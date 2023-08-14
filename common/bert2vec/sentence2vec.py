
from pybert.configs.basic_config import config

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    XLNetConfig,
    XLNetModel,
    XLNetTokenizer,
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer
)


MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, RobertaModel, XLNetTokenizer),
    "roberta": (RobertaConfig, XLNetModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
}


# model_type = 'albert'
# config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

config_class = BertConfig.from_pretrained(config['bert_model_dir'])
tokenizer = BertTokenizer.from_pretrained(config['bert_model_dir'])
model = BertModel.from_pretrained(config['bert_model_dir'])

print(config_class)
print(tokenizer)
print(model)
