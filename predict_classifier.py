from common.configs import get_config
from common.utils import read_json
from pyclassifier.model.AlBertForSequenceClassification import AlBertForSequenceClassification
from pyclassifier.model.BertForSequenceClassification import BertForSequenceClassification
from pyclassifier.model.XlNetForSequenceClassification import XlNetForSequenceClassification
from pyclassifier.test.predictor import Pred_sentence

TASK = 'classifier'
MODEL = 'xlnet'

config = get_config(TASK)
# 自动创建albert bert xlnet 输入目录
for dir in config['dir_list']:
    config[dir] = config[dir] / MODEL

label_list = read_json('label_list_{}.json'.format(MODEL), path=config['label_list'])

eval_max_seq_len = 16
train_batch_size = 8
n_gpu = '0'

assert MODEL in ['albert','bert','xlnet','roberta']
if MODEL == 'albert':
    model = AlBertForSequenceClassification.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

elif MODEL == 'bert' or MODEL == 'roberta':
    model = BertForSequenceClassification.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

else:
    model = XlNetForSequenceClassification.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))


Pred = Pred_sentence(model_type=MODEL,vocab_file=str(config['{}_vocab_path'.format(MODEL)]), model=model, max_seq_len=eval_max_seq_len,label_list=label_list,spm_model_file=None, do_lower_case=True, n_gpu=n_gpu)


indexs = Pred.get_max_indexs(MODEL,['我是罗俊宇','我不是罗俊宇'])

print('indexs',indexs)

# for i in range(10):
#     result1 = Pred.get_labels(MODEL,'我是罗俊宇')
#     print(result1)
#     result2 = Pred.get_labels(MODEL,'我不是罗俊宇')
#     print(result2)