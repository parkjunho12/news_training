#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import nltk
import re
import json
import os
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, LSTM
from keras.layers import TimeDistributed


# In[2]:


def read_data(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        file = f.read()
        # txt 파일의 헤더(id document label)는 제외하기
        file = file.replace('\n', ' ')
        file = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>▲�`\'…》]', '', file)
        data = file.split("다음 기사에요")
        data = data[1:]
    return data


# In[3]:


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    okt = Okt()
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


# In[4]:


if __name__ == '__main__':
    train_data = read_data('../datas/NEWS.txt')
    test_data = read_data('../datas/NEWS_SUM.txt')


# In[5]:


def reviewTokenize(datas):
    for index, data in enumerate(datas):
        hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
        datas[index] = hangul.sub("", data)
    return datas


# In[6]:


train_data = reviewTokenize(train_data)
test_data = reviewTokenize(test_data)


# In[7]:


print(train_data)


# In[8]:


train_docs = [(tokenize(data), tokenize(test_data[index])) for index, data in enumerate(train_data)]


# In[9]:


for doc in train_docs:
    print(doc)


# In[10]:


def make_json_file(train_docs):
        if os.path.isfile('train_docs.json'):
            with open('../datas/train_docs.json', encoding="utf-8") as f:
                train_docs = json.load(f)
        else:
            train_docs = [(tokenize(data), tokenize(test_data[index])) for index, data in enumerate(train_data)]
            # JSON 파일로 저장
            with open('../datas/train_docs.json', 'w', encoding="utf-8") as make_file:
                json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
        return train_docs


# In[11]:


train_docs = make_json_file(train_docs)


# In[23]:


def make_selected_words(train_docs):
    tokens = [t for d in train_docs for t in d[0]]
    text = nltk.Text(tokens, name='NMSC')
    selected_words = [f[0] for f in text.vocab().most_common(10000)]
    return selected_words


# In[24]:


selected_words = make_selected_words(train_docs)


# In[25]:


def change_frequency(docs):
    train_x = [term_frequency(d) for d, _ in docs]
    train_y = [term_frequency(c) for _, c in docs]
    return train_x, train_y


# In[26]:


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


# In[27]:


train_x, train_y = change_frequency(train_docs)


# In[28]:


for x in train_y:
    print(x)


# In[45]:


def modeling(x_train, y_train):
    model = models.Sequential()
    model.add(LSTM(128)) # try using a GRU instead, for fun
    model.add(LSTM(128)) # try using a GRU instead, for fun
    model.add(Dense(23))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    model.fit(x_train, y_train, epochs=10, batch_size=512)
    return model


# In[30]:


def array_to_float(data):
    x_train = np.asarray(data, dtype='float32').astype('float32')
    return x_train


# In[31]:


array_to_float(train_y)


# In[46]:


model = modeling(array_to_float(train_x), array_to_float(train_y))


# In[ ]:




