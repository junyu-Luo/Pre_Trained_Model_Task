r"""Functional interface"""
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import f1_score, classification_report
from sklearn import metrics

__call__ = ['Accuracy','AUC','F1Score','EntityScore','ClassReport','MultiLabelReport','AccuracyThresh']

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class Accuracy(Metric):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    Examples:
        >>> metric = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK
        self.reset()

    def __call__(self, logits, target):
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct_k = correct[:self.topK].view(-1).float().sum(0)
        self.total = target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k)  / self.total

    def name(self):
        return 'accuracy'


class AccuracyThresh(Metric):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    Example:
        >>> metric = AccuracyThresh(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self,thresh = 0.5):
        super(AccuracyThresh,self).__init__()
        self.thresh = thresh
        self.reset()

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid()
        self.y_true = target

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        # data_size = self.y_pred.size(0)
        # # print('self.y_pred',self.y_pred)
        # # print('self.y_true',self.y_true)
        # acc = np.mean(((self.y_pred>self.thresh)==self.y_true.bool()).float().cpu().numpy(), axis=1).sum()
        # return acc / data_size
        pred = np.array([np.argmax(x) for x in self.y_pred.cpu().detach().numpy()])
        true = np.array([np.argmax(x) for x in self.y_true.cpu().detach().numpy()])
        return accuracy_score(y_true=true,y_pred=pred)

    def name(self):
        return 'accuracy'


class AUC(Metric):
    '''
    AUC score
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = AUC(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self,task_type = 'binary',average = 'binary'):
        super(AUC, self).__init__()

        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average

    def __call__(self,logits,target):
        '''
        计算整个结果
        '''
        if self.task_type == 'binary':
            self.y_prob = logits.sigmoid().data.cpu().numpy()
        else:
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def value(self):
        '''
        计算指标得分
        '''
        # print([[label_list[int(y)] for y in np.where(x.label == 1)][0] for x in self.y_true])
        auc = roc_auc_score(y_score=[np.argmax(x) for x in self.y_prob], y_true=[np.argmax(x) for x in self.y_true], average=self.average)
        return auc

    def name(self):
        return 'auc'

class F1Score(Metric):
    '''
    F1 Score
    binary:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = F1Score(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(F1Score).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        '''
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们队Thresh进行优化
        :return:
        '''
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        '''
        计算整个结果
        :return:
        '''
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh ).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        '''
         计算指标得分
         '''
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'

class ClassReport(Metric):
    '''
    class report
    '''
    def __init__(self,target_names = None):
        super(ClassReport).__init__()
        self.target_names = target_names

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        '''
        计算指标得分
        '''
        # print('self.y_true',self.y_true)
        # print('self.y_pred',self.y_pred)
        # print('y_true',[np.argmax(x) for x in self.y_true])
        # print('y_pred',[np.argmax(x) for x in self.y_pred])
        score = classification_report(y_true = [np.argmax(x) for x in self.y_true],
                                      y_pred = [np.argmax(x) for x in self.y_pred],
                                      target_names=self.target_names)
        print(f"\n\n classification report: {score}")


    def __call__(self,logits,target):
        # _, y_pred = torch.max(logits.data, 1)
        self.y_pred = logits.cpu().numpy()
        self.y_true = target.cpu().numpy()

    def name(self):
        return "class_report"

class MultiLabelReport(Metric):
    '''
    multi label report
    '''
    def __init__(self,id2label = None):
        super(MultiLabelReport).__init__()
        self.id2label = id2label

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def __call__(self,logits,target):
        self.y_pred = logits.cpu().numpy()
        # self.y_prob = logits.sigmoid().data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        '''
        计算指标得分
        '''
        pred = np.array([np.argmax(x) for x in self.y_pred])
        true = np.array([np.argmax(x) for x in self.y_true])
        for i, label in self.id2label.items():
            auc = roc_auc_score(y_score=pred==i, y_true=true==i)
            print(f"label:{label} - auc: {auc:.4f}")

    def name(self):
        return "multilabel_report"


def multi_dev_metrics(real,pred,not_magor_labels):
    assert len(real) == len(pred)
    real_result = []
    pred_result = []
    for i in range(len(real)):
        if len(real[i]) == len(pred[i]):
            real_result.extend(sorted(real[i]))
            pred_result.extend(sorted(pred[i]))
        elif len(real[i]) > len(pred[i]):
            real_tmp = real[i]  # ['4', '2', '1']
            pred_tmp = pred[i]  # ['4','缺失','缺失']
            for p in pred_tmp:
                if p in real_tmp:
                    real_result.append(p)
                    pred_result.append(p)
                    real_tmp.remove(p)
                    pred_tmp.remove(p)
            for _ in range(len(real[i]) - len(pred[i])):
                pred_tmp.append('缺失')
            real_result.extend(real_tmp)
            pred_result.extend(pred_tmp)
        else:
            real_tmp = real[i]  # ['4', '3','多余']
            pred_tmp = pred[i]  # ['4', '2', '1']
            for r in real_tmp:
                if r in pred_tmp:
                    real_result.append(r)
                    pred_result.append(r)
                    real_tmp.remove(r)
                    pred_tmp.remove(r)
            for _ in range(len(pred[i]) - len(real[i])):
                real_tmp.append('多余')
            real_result.extend(real_tmp)
            pred_result.extend(pred_tmp)
    labels = []
    labels.extend(real_result)
    labels.extend(pred_result)
    magor_labels = list(set(labels))
    if '多余' in magor_labels:
        magor_labels.remove('多余')
    if '缺失' in magor_labels:
        magor_labels.remove('缺失')
    for label in not_magor_labels:
        if label in magor_labels:
            magor_labels.remove(label)
    return metrics.precision_score(real_result, pred_result, labels=magor_labels, average='macro')  # 指定特定分类标签的精确率


def multi_train_metrics(real,pred,label_list):
    assert '缺失' not in label_list
    assert '多余' not in label_list
    assert len(real)==len(pred)
    real_result = []
    pred_result = []
    for i in range(len(real)):
        if len(real[i]) == len(pred[i]):
            real_result.extend(sorted(real[i]))
            pred_result.extend(sorted(pred[i]))
        elif len(real[i]) > len(pred[i]):
            real_tmp = real[i]  #['4', '2', '1']
            pred_tmp = pred[i]  #['4','缺失','缺失']
            for p in pred_tmp:
                if p in real_tmp:
                    real_result.append(p)
                    pred_result.append(p)
                    real_tmp.remove(p)
                    pred_tmp.remove(p)
            for _ in range(len(real[i])-len(pred[i])):
                pred_tmp.append('缺失')
            real_result.extend(real_tmp)
            pred_result.extend(pred_tmp)
        else:
            real_tmp = real[i]  # ['4', '3','多余']
            pred_tmp = pred[i]  # ['4', '2', '1']
            for r in real_tmp:
                if r in pred_tmp:
                    real_result.append(r)
                    pred_result.append(r)
                    real_tmp.remove(r)
                    pred_tmp.remove(r)
            for _ in range(len(pred[i])-len(real[i])):
                real_tmp.append('多余')
            real_result.extend(real_tmp)
            pred_result.extend(pred_tmp)
    # label_list.extend(['缺失', '多余'])
    target_names = []
    target_names.extend(real_result)
    target_names.extend(pred_result)
    # print('real_result',real_result)
    # print('pred_result',pred_result)
    # print('label_list',label_list)
    return classification_report(real_result, pred_result, target_names=list(set(target_names)))

# if __name__ == '__main__':
#     label_list = ['1', '2', '3', '4']
#     real = [['1'], ['3', '1'], ['4', '2', '1'], ['3','4',], ['4']]
#     pred = [['1'], ['1', '3'], ['4'], ['3','4', '2', '1'], ['3']]
#
#     print(multi_train_metrics(real,pred,label_list))