import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import nltk

def read_data(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        file = f.read()
        # txt 파일의 헤더(id document label)는 제외하기
        file = file.replace('\n', ' ')
        data = file.split("다음 기사에요")
        data = data[1:]
    return data


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    okt = Okt()
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def reviewTokenize(datas):
    for index, data in enumerate(datas):
        datas[index] = data.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    return datas


if __name__ == '__main__':
    train_data = read_data('../datas/NEWS.txt')
    test_data = read_data('../datas/NEWS_SUM.txt')

    train_docs = [(tokenize(data[index]), tokenize(test_data[index])) for index, data in enumerate(train_data)]

    for index, data in enumerate(train_data):
        print(data)
        print("\n")

    for data in train_docs:
        print(data)
        print("\n")