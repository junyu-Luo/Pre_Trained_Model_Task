from pathlib import Path
BASE_DIR = Path('./')
DATASET_DIR = Path('dataset/classifier_multi_task')
OUTPUT_DIR = Path('output/classifier_multi_task')
PRETRAIN_MODEL_DIR = Path('model_pretrain')
config = {
    'data_dir': DATASET_DIR,
    'output_dir': OUTPUT_DIR,
    'dir_list':['checkpoint_dir', 'log_dir', 'figure_dir', 'cache_dir', 'result', 'label_list', 'writer_dir'],
    'checkpoint_dir': OUTPUT_DIR / "checkpoints",
    'log_dir': OUTPUT_DIR / 'log',
    'figure_dir': OUTPUT_DIR / "figure",
    'cache_dir': OUTPUT_DIR / 'data_cache/',
    'result': OUTPUT_DIR / "result",
    'label_list': OUTPUT_DIR / "label_list",
    'writer_dir': OUTPUT_DIR / "TsBoard",

# 默认使用base
    'bert_vocab_path': PRETRAIN_MODEL_DIR / 'bert/chinese_L-12_H-768_A-12/vocab.txt',
    'bert_config_file': PRETRAIN_MODEL_DIR / 'bert/chinese_L-12_H-768_A-12/bert_config.json',
    'bert_model_dir': PRETRAIN_MODEL_DIR / 'bert/chinese_L-12_H-768_A-12',

    'xlnet_vocab_path': PRETRAIN_MODEL_DIR / 'xlnet/base-cased/spiece.model',
    'xlnet_config_file': PRETRAIN_MODEL_DIR / 'xlnet/base-cased/config.json',
    'xlnet_model_dir': PRETRAIN_MODEL_DIR / 'xlnet/base-cased',

    'roberta_vocab_path': PRETRAIN_MODEL_DIR / 'roberta/RoBERTa-wwm-ext/vocab.txt',
    'roberta_config_file': PRETRAIN_MODEL_DIR / 'roberta/RoBERTa-wwm-ext/bert_config.json',
    'roberta_model_dir': PRETRAIN_MODEL_DIR / 'roberta/RoBERTa-wwm-ext',

    'albert_spm_model': PRETRAIN_MODEL_DIR / 'albert/albert-base/30k-clean.model',
    'albert_vocab_path': PRETRAIN_MODEL_DIR / 'albert/albert-base/vocab.txt',
    'albert_config_file': PRETRAIN_MODEL_DIR / 'albert/albert-base/config.json',
    'albert_model_dir': PRETRAIN_MODEL_DIR / 'albert/albert-base',


}

