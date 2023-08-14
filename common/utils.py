import json
import pickle
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import shutil,os

def write_file(file, data, path='data', mode='a+', encoding='utf-8'):
    writefile = open(os.path.join(path, file), mode, encoding=encoding)
    writefile.write(data + '\n')
    writefile.close()


def write_lis(file, data_list, path='data', mode='a+', encoding='utf-8'):
    with open(os.path.join(path, file), mode, encoding=encoding) as f:
        for line in data_list:
            f.write(line + '\n')
    f.close()


def read_in_block(file, path='data'):
    with open(os.path.join(path, file), "r", encoding="utf-8") as f:
        while True:
            block = f.readline()  # 每次读取固定长度到内存缓冲区
            if block:
                yield block
            else:
                return  # 如果读取到文件末尾，则退出


def save_json(file, content, path='js_file'):
    f = open(os.path.join(path, file), 'a+', encoding='utf-8')
    json_str = json.dumps(content, ensure_ascii=False)
    f.write(json_str)
    f.close()


def read_json(file, encoding='utf-8', path='js_file'):
    data = open(os.path.join(path, file), encoding=encoding).read()
    return json.loads(data)


def save_pkl(file, content, path='pkl_file'):
    f = open(os.path.join(path, file), 'wb')
    pickle.dump(content, f)
    f.close()


def load_pkl(file, path='pkl_file'):
    with open(os.path.join(path, file), 'rb') as f:
        return pickle.load(f)


def distrubute(length_list):
    """
    打印句子分布
    :param length_list: list
    :return: 无
    """
    length_np = np.array(length_list)
    # 均值
    print('均值', np.mean(length_np))
    # 中位数
    print('中位数', np.median(length_np))
    # 返回众数
    print('众数', np.argmax(np.bincount(length_np)))


def xlsx2json(xlsx_file, js_filename, path='data'):
    def to_json_str(df, orient='index'):
        return df.to_json(orient=orient, force_ascii=False)

    def to_json_obj(df, orient='index'):
        df_json = df.to_json(orient=orient, force_ascii=False)
        result = []
        json_obj = json.loads(df_json)
        for i in range(len(json_obj)):
            result.append(json_obj.get(str(i)))
        return result

    df = pd.read_excel(os.path.join(path, xlsx_file))
    # json_str = to_json_str(df)
    save_json(js_filename, to_json_obj(df))


def json2xlsx(js_file, xlsx_filename, path='data'):
    data = open(os.path.join(path, js_file), encoding="utf-8").read()
    data_list = json.loads(data)
    df = json_normalize(data_list)
    df.to_excel(xlsx_filename)
    # df.to_csv(xlsx_filename)


def json2csv(js_file, xlsx_filename, path='data'):
    data = open(os.path.join(path, js_file), encoding="utf-8").read()
    data_list = json.loads(data)
    df = json_normalize(data_list)
    df.to_csv(xlsx_filename)


def clean_dir(DIR):
    if os.path.exists(DIR):
        shutil.rmtree(DIR)
        os.mkdir(DIR)
    else:
        print('dir not exists and now mkdir new')
        os.mkdir(DIR)