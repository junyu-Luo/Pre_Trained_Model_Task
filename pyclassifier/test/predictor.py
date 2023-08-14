#encoding:utf-8
import torch
import numpy as np
from common.tools import model_device
from ..callback.progressbar import ProgressBar
from pyclassifier_multi.model.albert.tokenization_albert import FullTokenizer
from transformers import BertTokenizer
from transformers import XLNetTokenizer

class Predictor(object):
    def __init__(self,model,logger,n_gpu):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        
    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        all_logits = None
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()
                # print('input_ids', input_ids)
                # print('input_mask', input_mask)
                # print('segment_ids', segment_ids)
                # print('label_ids', label_ids)
                # print('logits', logits)
            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
            pbar(step=step)
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_logits

class Pred_sentence(object):
    def __init__(self,model_type,vocab_file,model,max_seq_len,label_list,spm_model_file=None,do_lower_case=True,n_gpu='0'):
        if model_type == 'albert':
            self.albert_tokenizer = FullTokenizer(vocab_file=vocab_file,do_lower_case=do_lower_case, spm_model_file=spm_model_file)
        if model_type == 'bert' or model_type == 'roberta':
            self.bert_tokenizer = BertTokenizer(vocab_file=vocab_file,do_lower_case=do_lower_case)
        if model_type == 'xlnet':
            self.xlnet_tokenizer = XLNetTokenizer(vocab_file=vocab_file,do_lower_case=do_lower_case)
        self.model = model
        self.model, self.device = model_device(n_gpu=n_gpu, model=self.model)
        self.max_seq_len = max_seq_len
        self.label_list = label_list

    def get_label(self,model_type,sentence):
        if model_type == 'xlnet':
            pad_token = self.xlnet_tokenizer.convert_tokens_to_ids([self.xlnet_tokenizer.pad_token])[0]
            cls_token = self.xlnet_tokenizer.cls_token
            sep_token = self.xlnet_tokenizer.sep_token
            cls_token_segment_id = 2
            pad_token_segment_id = 4
            tokens = self.xlnet_tokenizer.tokenize(sentence)
            if len(tokens) > self.max_seq_len - 2:
                tokens = tokens[:self.max_seq_len - 2]
            tokens = tokens + [sep_token]
            segment_ids = [0] * len(tokens)
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]

            input_ids = self.xlnet_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [0] * len(input_ids)
            padding_len = self.max_seq_len - len(input_ids)

            # pad on the left for xlnet
            input_ids = input_ids + ([pad_token] * padding_len)
            input_mask = input_mask + ([1] * padding_len)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_len)

            assert len(input_ids) == len(input_mask) == len(segment_ids) == self.max_seq_len

            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long)
            # print(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
                logits = logits.sigmoid()
            logits = logits.detach().cpu().numpy()
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

            index = np.argmax(logits[0])
            return self.label_list[index]

        else:
            tokens = self.bert_tokenizer.tokenize(sentence)
            if len(tokens) > self.max_seq_len - 2:
                tokens = tokens[:self.max_seq_len - 2]

            tokens = ['[CLS]'] + tokens + ['[SEP]']
            segment_ids = [0] * len(tokens)
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (self.max_seq_len - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == len(input_mask) == len(segment_ids) == self.max_seq_len

            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long)
            # print(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
                logits = logits.sigmoid()
            logits = logits.detach().cpu().numpy()

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

            index = np.argmax(logits[0])
            return self.label_list[index]

    def get_max_indexs(self,model_type,sentences):
        if model_type == 'xlnet':
            pad_token = self.xlnet_tokenizer.convert_tokens_to_ids([self.xlnet_tokenizer.pad_token])[0]
            cls_token = self.xlnet_tokenizer.cls_token
            sep_token = self.xlnet_tokenizer.sep_token
            cls_token_segment_id = 2
            pad_token_segment_id = 4
            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []
            for sentence in sentences:
                tokens = self.xlnet_tokenizer.tokenize(sentence)
                if len(tokens) > self.max_seq_len - 2:
                    tokens = tokens[:self.max_seq_len - 2]
                tokens = tokens + [sep_token]
                segment_ids = [0] * len(tokens)
                tokens += [cls_token]
                segment_ids += [cls_token_segment_id]

                input_ids = self.xlnet_tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [0] * len(input_ids)
                padding_len = self.max_seq_len - len(input_ids)

                # pad on the left for xlnet
                input_ids = input_ids + ([pad_token] * padding_len)
                input_mask = input_mask + ([1] * padding_len)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_len)

                assert len(input_ids) == len(input_mask) == len(segment_ids) == self.max_seq_len
                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)


            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            input_mask = torch.tensor(input_mask_list, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids_list, dtype=torch.long)
            print('input_ids',input_ids)
            print('input_mask',input_mask)
            print('segment_ids',segment_ids)
            # print(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
                logits = logits.sigmoid()
            logits = logits.detach().cpu().numpy()
            print('logits',logits)
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

            indexs = logits.argmax(axis=0)
            # print('indexs',indexs)
            return self.label_list,indexs

        else:
            input_ids_list = []
            input_mask_list = []
            segment_ids_list = []
            for sentence in sentences:
                tokens = self.bert_tokenizer.tokenize(sentence)
                if len(tokens) > self.max_seq_len - 2:
                    tokens = tokens[:self.max_seq_len - 2]

                tokens = ['[CLS]'] + tokens + ['[SEP]']
                segment_ids = [0] * len(tokens)
                input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [0] * (self.max_seq_len - len(input_ids))

                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == len(input_mask) == len(segment_ids) == self.max_seq_len
                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)

            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            input_mask = torch.tensor(input_mask_list, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids_list, dtype=torch.long)
            # print(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
                logits = logits.sigmoid()
            logits = logits.detach().cpu().numpy()

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

            indexs = logits.argmax(axis=0)
            return indexs
            # return self.label_list[index]

    def albert_label(self,sentence):
        tokens = self.albert_tokenizer.tokenize(sentence)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        segment_ids = [0] * len(tokens)
        input_ids = self.albert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_len - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == len(input_mask) == len(segment_ids) == self.max_seq_len

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_mask = torch.tensor([input_mask], dtype=torch.long)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long)
        # print(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
            logits = logits.sigmoid()
        logits = logits.detach().cpu().numpy()

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        index = np.argmax(logits[0])
        return self.label_list[index]

    def bert_label(self,sentence):
        tokens = self.bert_tokenizer.tokenize(sentence)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        segment_ids = [0] * len(tokens)
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_len - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == len(input_mask) == len(segment_ids) == self.max_seq_len

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_mask = torch.tensor([input_mask], dtype=torch.long)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
            logits = logits.sigmoid()
        logits = logits.detach().cpu().numpy()

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        index = np.argmax(logits[0])
        return self.label_list[index]

    def xlnet_label(self,sentence):
        pad_token = self.xlnet_tokenizer.convert_tokens_to_ids([self.xlnet_tokenizer.pad_token])[0]
        cls_token = self.xlnet_tokenizer.cls_token
        sep_token = self.xlnet_tokenizer.sep_token
        cls_token_segment_id = 2
        pad_token_segment_id = 4
        tokens = self.xlnet_tokenizer.tokenize(sentence)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        tokens = tokens + [sep_token]
        segment_ids = [0] * len(tokens)
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]

        input_ids = self.xlnet_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [0] * len(input_ids)
        padding_len = self.max_seq_len - len(input_ids)

        # pad on the left for xlnet
        input_ids = input_ids + ([pad_token] * padding_len)
        input_mask = input_mask + ([1] * padding_len)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_len)

        assert len(input_ids) == len(input_mask) == len(segment_ids) == self.max_seq_len

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_mask = torch.tensor([input_mask], dtype=torch.long)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long)
        # print(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids.to(self.device), segment_ids.to(self.device), input_mask.to(self.device))
            logits = logits.sigmoid()
        logits = logits.detach().cpu().numpy()

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        index = np.argmax(logits[0])
        return self.label_list[index]



