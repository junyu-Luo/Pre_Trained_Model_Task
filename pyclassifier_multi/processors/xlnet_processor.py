import csv
import os
import torch
import numpy as np
from common.tools import load_pickle
from common.tools import logger
from ..callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset
from transformers import XLNetTokenizer
from common.utils import save_json,read_json,read_in_block

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_input_lens = map(torch.stack, zip(*batch))
    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self,input_ids,input_mask,segment_ids,label_id,input_len):
        self.input_ids   = input_ids
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id
        self.input_len = input_len

class XlnetProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,vocab_path,do_lower_case):
        self.tokenizer = XLNetTokenizer(vocab_path,do_lower_case)

    def get_train(self, data_dir,data_name="train.tsv"):
        """Gets a collection of `InputExample`s for the train set."""
        lines = self._read_tsv(os.path.join(data_dir, data_name))
        # self.labels.extend([x[0] for x in lines])
        # return self._create_examples(lines, "train")
        return lines

    def get_dev(self, data_dir,data_name="dev.tsv"):
        """Gets a collection of `InputExample`s for the dev set."""
        lines = self._read_tsv(os.path.join(data_dir, data_name))
        # self.labels.extend([x[0] for x in lines])
        # return self._create_examples(lines, "dev")
        return lines

    def get_test(self, data_dir, data_name="test.tsv"):
        """Gets a collection of `InputExample`s for the dev set."""
        lines = self._read_tsv(os.path.join(data_dir, data_name))
        # self.labels.extend([x[0] for x in lines])
        # return self._create_examples(lines, "dev")
        return lines

    def get_labels(self, config, split_symbol='\t',multi_symbol=','):
        """See base class."""
        if os.path.exists(config['label_list'] / 'label_list_xlnet.json'):
            label_list = read_json('label_list_xlnet.json', path=config['label_list'])
            return label_list
        else:
            labels = []
            for line in read_in_block("train.tsv", path=config['data_dir']):
                print('line', line)
                if len(line) > 0:
                    for i in line.split(split_symbol)[0].split(multi_symbol):
                        labels.append(i)
            label_list = sorted(list(set(labels)))
            save_json('label_list_xlnet.json', label_list, path=config['label_list'])
            return label_list

    # def get_labels(self):
    #     """Gets the list of labels for this data set."""
    #     return ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

    # @classmethod
    # def _read_tsv(cls, input_file, quotechar=None):
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r", encoding="utf-8-sig") as f:
    #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #         lines = []
    #         for line in reader:
    #             if len(line) > 0:
    #                 lines.append(line[0].split('    '))
    #         return lines

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        lines = []
        for line in read_in_block(input_file, path=''):
            line = line.replace('\n', '').split('\t')
            if len(line) == 2:
                lines.append(line)
        return lines

    @classmethod
    def read_data(cls, input_file,quotechar = None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines


    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self,lines,config,example_type,cached_examples_file,multi_symbol=','):
        '''
        Creates examples for data
        '''
        label_list = self.get_labels(config)
        label_np = np.array(label_list)
        pbar = ProgressBar(n_total=len(lines), desc='create examples')
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i, line in enumerate(lines):
                if len(line) == 2:
                    guid = '%s-%d' % (example_type, i)
                    text_a = line[1]
                    labels = line[0]
                    input_label = np.zeros(len(label_list))
                    assert isinstance(labels, str)
                    for label in labels.split(multi_symbol):
                        input_label[np.where(label_np == label)] = 1
                    if i == 0:
                        print('input_label', input_label)

                    text_b = None
                    example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=input_label)
                    examples.append(example)
                    pbar(step=i)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features(self, examples, max_seq_len, cached_features_file):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        这个是xlnet 不是bert的序列 xlnet 是 CLS在最后
        '''
        # Load data features from cache or dataset file
        pbar = ProgressBar(n_total=len(examples), desc='create features')
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
            cls_token = self.tokenizer.cls_token
            sep_token = self.tokenizer.sep_token
            cls_token_segment_id = 2
            pad_token_segment_id = 4

            for ex_id, example in enumerate(examples):
                tokens_a = self.tokenizer.tokenize(example.text_a)
                tokens_b = None
                label_id = example.label

                if example.text_b:
                    tokens_b = self.tokenizer.tokenize(example.text_b)
                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    self.truncate_seq_pair(tokens_a, tokens_b, max_length=max_seq_len - 3)
                else:
                    # Account for [CLS] and [SEP] with '-2'
                    if len(tokens_a) > max_seq_len - 2:
                        tokens_a = tokens_a[:max_seq_len - 2]

                # xlnet has a cls token at the end
                tokens = tokens_a + [sep_token]
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [1] * (len(tokens_b) + 1)
                tokens += [cls_token]
                segment_ids += [cls_token_segment_id]

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [0] * len(input_ids)
                input_len = len(input_ids)
                padding_len = max_seq_len - len(input_ids)

                # pad on the left for xlnet
                input_ids = input_ids + ([pad_token] * padding_len)
                input_mask = input_mask + ([1] * padding_len)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_len)

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {example.guid}" % ())
                    logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")

                feature = InputFeature(input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       label_id=label_id,
                                       input_len=input_len)
                features.append(feature)
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self,features,is_sorted = False):
        # Convert to Tensors and build dataset
        if is_sorted:
            logger.info("sorted data by th length of input")
            features = sorted(features,key=lambda x:x.input_len,reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features],dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_input_lens)
        return dataset

