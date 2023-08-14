import os
from pathlib import Path

# BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BASE_DIR = 'pybert'
config = {
    'raw_data_path': os.path.join(BASE_DIR, 'dataset', 'train_sample.csv'),
    'test_path': os.path.join(BASE_DIR, 'dataset', 'test.tsv'),

    'data_dir': os.path.join(BASE_DIR, 'dataset'),
    'log_dir': os.path.join(BASE_DIR, 'output', 'log'),
    'writer_dir': os.path.join(BASE_DIR, 'output', 'TSboard'),
    'figure_dir': os.path.join(BASE_DIR, 'output', 'figure'),
    'checkpoint_dir': os.path.join(BASE_DIR, 'output', 'checkpoints'),
    'cache_dir': os.path.join(BASE_DIR, 'model'),
    'result': os.path.join(BASE_DIR, 'output', 'result'),

    # 默认使用base
    'bert_vocab_path': os.path.join(BASE_DIR, 'pretrain', 'bert', 'chinese_L-12_H-768_A-12', 'vocab.txt'),
    'bert_config_file': os.path.join(BASE_DIR, 'pretrain', 'bert', 'chinese_L-12_H-768_A-12', 'bert_config.json'),
    'bert_model_dir': os.path.join(BASE_DIR, 'pretrain', 'bert', 'chinese_L-12_H-768_A-12'),

    'xlnet_vocab_path': os.path.join(BASE_DIR, 'pretrain', 'xlnet', 'base-cased', 'vocab.txt'),
    'xlnet_config_file': os.path.join(BASE_DIR, 'pretrain', 'xlnet', 'base-cased', 'config.json'),
    'xlnet_model_dir': os.path.join(BASE_DIR, 'pretrain', 'xlnet', 'base-cased'),

    'roberta_vocab_path': os.path.join(BASE_DIR, 'pretrain', 'roberta', 'RoBERTa-wwm-ext', 'vocab.txt'),
    'roberta_config_file': os.path.join(BASE_DIR, 'pretrain', 'roberta', 'RoBERTa-wwm-ext', 'bert_config.json'),
    'roberta_model_dir': os.path.join(BASE_DIR, 'pretrain', 'roberta', 'RoBERTa-wwm-ext'),

    'albert_spm_model': os.path.join(BASE_DIR, 'pretrain', 'albert', 'albert-base', '30k-clean.model'),
    'albert_vocab_path': os.path.join(BASE_DIR, 'pretrain', 'albert', 'albert-base', 'vocab.txt'),
    'albert_config_file': os.path.join(BASE_DIR, 'pretrain', 'albert', 'albert-base', 'config.json'),
    'albert_model_dir': os.path.join(BASE_DIR, 'pretrain', 'albert', 'albert-base'),

    # 其他选择

}