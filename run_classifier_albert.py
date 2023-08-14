import torch
import time,os
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pyclassifier.train.losses import BCEWithLogLoss
from pyclassifier.train.trainer import Trainer
from torch.utils.data import DataLoader
from common.tools import init_logger, logger
from common.tools import seed_everything
from common.configs import get_config
from pyclassifier.processors.albert_processor import AlbertProcessor,collate_fn
from pyclassifier.model.AlBertForSequenceClassification import AlBertForSequenceClassification
from pyclassifier.callback.modelcheckpoint import ModelCheckpoint
from pyclassifier.callback.trainingmonitor import TrainingMonitor
from pyclassifier.train.metrics import AUC, AccuracyThresh, MultiLabelReport
from pyclassifier.callback.optimizater.adamw import AdamW
from pyclassifier.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
warnings.filterwarnings("ignore")
config = get_config('classifier')

def run_train(args):
    # --------- data
    processor = AlbertProcessor(spm_model_file=None, do_lower_case=args.do_lower_case,
                                vocab_file=config['albert_vocab_path'])
    label_list = processor.get_labels(data_dir=config['data_dir'])
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'],data_name='train.tsv')
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'cache_dir'] / f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'cache_dir'] / "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    valid_data = processor.get_dev(config['data_dir'],data_name='dev.tsv')
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=config[
                                                'cache_dir'] / f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                'cache_dir'] / "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)

    # ------- model
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = AlBertForSequenceClassification.from_pretrained(args.resume_path, num_labels=len(label_list))
    else:
        model = AlBertForSequenceClassification.from_pretrained(config['albert_model_dir'], num_labels=len(label_list))
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(args= args,model=model,logger=logger,criterion=BCEWithLogLoss(),optimizer=optimizer,
                      scheduler=scheduler,early_stopping=None,training_monitor=train_monitor,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      epoch_metrics=[AUC(average='micro', task_type='binary'),
                                     MultiLabelReport(id2label=id2label)])
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)

def run_test(args):
    from pyclassifier.test.predictor import Predictor

    processor = AlbertProcessor(spm_model_file=None, do_lower_case=args.do_lower_case,
                                vocab_file=config['albert_vocab_path'])
    label_list = processor.get_labels(data_dir=config['data_dir'])
    test_data = processor.get_test(config['data_dir'], data_name='test.tsv')

    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=config[
                                             'cache_dir'] / f"cached_test_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                              'cache_dir'] / "cached_test_features_{}_{}".format(
                                                  args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,
                                 collate_fn=collate_fn)
    model = AlBertForSequenceClassification.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,logger=logger,n_gpu=args.n_gpu)
    result = predictor.predict(data=test_dataloader)
    # print(result)
    from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
    pred = [label_list[np.argmax(x)] for x in result]
    real = [[label_list[int(y)] for y in np.where(x.label == 1)][0] for x in test_examples]

    report = classification_report(real, pred, labels=label_list)
    logger.info('acc is ' + str(accuracy_score(real, pred)))
    logger.info(str(report))

def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='albert', type=str)

    parser.add_argument("--do_train", default=True, action='store_true')
    parser.add_argument("--do_test", default=True, action='store_true')
    parser.add_argument("--save_best", default=True, action='store_true')
    parser.add_argument("--do_lower_case", default=True, action='store_true')
    parser.add_argument('--data_name', default='', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--overwrite", default=False, action='store_true')

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=0, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=16, type=int)
    parser.add_argument("--eval_max_seq_len", default=16, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    # output 创建文件夹
    config['output_dir'].mkdir(exist_ok=True)

    # 是否覆盖，若覆盖全部删除掉
    if args.overwrite and args.do_train:
        from common.utils import clean_dir
        for dir in config['dir_list']:
            clean_dir(config[dir] / args.arch)

    # 自动创建albert bert xlnet 输入目录
    def auto_mkdir(dir_list):
        for dir in dir_list:
            config[dir].mkdir(exist_ok=True)
            config[dir] = config[dir] / args.arch
            config[dir].mkdir(exist_ok=True)
    auto_mkdir(config['dir_list'])

    init_logger(log_file= os.path.join(config['log_dir'],'{}.log'.format(str(time.strftime("%m%d_%H%M", time.localtime())))))

    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)


    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()
