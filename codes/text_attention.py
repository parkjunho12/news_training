import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from keras import layers, models
from keras import datasets
from keras import backend as K

import nltk
import re
import json
import os
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, LSTM
from keras.layers import TimeDistributed


def read_data(filename): #파일 읽어 오기
    with open(filename, 'r', encoding="utf-8") as f:
        file = f.read()
        # txt 파일의 헤더(id document label)는 제외하기
        file = file.replace('\n', ' ') # 엔터 제거
        file = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>▲�`\'…》]', '', file) # 특수 문자 제거
        data = file.split("다음 기사에요") # 기사 별로 담기
        data = data[1:] # label 제거
    return data


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    okt = Okt()
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def reviewTokenize(datas):
    for index, data in enumerate(datas):
        hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
        datas[index] = hangul.sub("", data)
    return datas


# 학습 정보
batch_size = 100
epochs = 24
latent_dim = 256
num_samples = 289

# 문장 벡터화
input_texts = read_data('../datas/NEWS.txt')
target_texts = read_data('../datas/NEWS_SUM.txt')
input_texts = reviewTokenize(input_texts)
target_texts = reviewTokenize(target_texts)
input_characters = set()
target_characters = set()


for input_text in input_texts:
    for word in tokenize(input_text):
        words = word.split('/')
        if words[0] not in input_characters:
            input_characters.add(words[0])
for target_text in target_texts:
    for word in tokenize(target_text):
        words = word.split('/')
        if words[0] not in input_characters:
            target_characters.add(words[0])

num_samples = len(input_texts)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', num_samples)
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)]
)
print(input_token_index)
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)]
)


def storeTrainData(data):
    X_train = []

    # 한글과 공백을 제외하고 모두 제거
    okt = Okt()
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    temp_X = okt.morphs(data, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제

    return temp_X


#학슴에 사용할 데이터를 담을 3차원 배열
encoder_input_data = np.zeros(
    (num_samples,
     max_encoder_seq_length,
     num_encoder_tokens),
    dtype='float32')

decoder_input_data = np.zeros(
    (num_samples,
     max_decoder_seq_length,
     num_decoder_tokens),
    dtype='float32')

decoder_target_data = np.zeros(
    (num_samples,
     max_decoder_seq_length,
     num_decoder_tokens),
    dtype='float32')

# 문장을 문자 단위로 원 핫 인코딩 하면서 학습용 데이터를 만듬
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(storeTrainData(input_text)):
        if char in input_token_index.keys():
            encoder_input_data[i, t, input_token_index[char]] = 1
    for t, char in enumerate(storeTrainData(target_text)):
        if char in target_token_index.keys():
            decoder_input_data[i, t, target_token_index[char]] = 1
    if t > 0:
        if char in target_token_index.keys():
            decoder_target_data[i, t-1, target_token_index[char]] = 1

reverse_input_char_index = dict( (i, char)
                                 for char, i in input_token_index.items())
reverse_target_char_index = dict( (i, char)
                                  for char, i in target_token_index.items())


def RepeatVectorLayer(rep, axis):
    return layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis), rep, axis),
                         lambda x: tuple((x[0],) + x[1:axis] + (rep,) + x[axis:]))


# 인코더 생성
encoder_inputs = layers.Input(shape=(max_encoder_seq_length, num_encoder_tokens))
encoder = layers.GRU(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)

# 디코더 생성.
decoder_inputs = layers.Input(shape=(max_decoder_seq_length, num_decoder_tokens))
decoder = layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder(decoder_inputs, initial_state=state_h)


# 어텐션 매커니즘.
repeat_d_layer = RepeatVectorLayer(max_encoder_seq_length, 2)
repeat_d = repeat_d_layer(decoder_outputs)
repeat_e_layer = RepeatVectorLayer(max_decoder_seq_length, 1)
repeat_e = repeat_e_layer(encoder_outputs)
concat_for_score_layer = layers.Concatenate(axis=-1)
concat_for_score = concat_for_score_layer([repeat_d, repeat_e])
dense1_t_score_layer = layers.Dense(latent_dim // 2, activation='tanh')
dense1_score_layer = layers.TimeDistributed(dense1_t_score_layer)
dense1_score = dense1_score_layer(concat_for_score)
dense2_t_score_layer = layers.Dense(1)
dense2_score_layer = layers.TimeDistributed(dense2_t_score_layer)
dense2_score = dense2_score_layer(dense1_score)
dense2_score = layers.Reshape((max_decoder_seq_length, max_encoder_seq_length))(dense2_score)
softmax_score_layer = layers.Softmax(axis=-1)
softmax_score = softmax_score_layer(dense2_score)
repeat_score_layer = RepeatVectorLayer(latent_dim, 2)
repeat_score = repeat_score_layer(softmax_score)
permute_e = layers.Permute((2, 1))(encoder_outputs)
repeat_e_layer = RepeatVectorLayer(max_decoder_seq_length, 1)
repeat_e = repeat_e_layer(permute_e)
attended_mat_layer = layers.Multiply()
attended_mat = attended_mat_layer([repeat_score, repeat_e])
context_layer = layers.Lambda(lambda x: K.sum(x, axis=-1), lambda x: tuple(x[:-1]))
context = context_layer(attended_mat)
concat_context_layer = layers.Concatenate(axis=-1)
concat_context = concat_context_layer([context, decoder_outputs])
attention_dense_output_layer = layers.Dense(latent_dim, activation='tanh')
attention_output_layer = layers.TimeDistributed(attention_dense_output_layer)
attention_output = attention_output_layer(concat_context)
decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(attention_output)

# 모델 생성
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
tb_hist = tf.keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True,
                                         write_images=True)
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=2)

