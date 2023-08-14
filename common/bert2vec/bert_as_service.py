# pip install bert-serving-server  # server
# pip install bert-serving-client  # client, independent of `bert-serving-server`

from bert_serving.client import BertClient
bc = BertClient(ip='xx.xx.xx.xx')  # ip address of the GPU machine
vec = bc.encode(['测试', '这是一个测试']) # show_tokens=True

# Start the BERT service
# bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4