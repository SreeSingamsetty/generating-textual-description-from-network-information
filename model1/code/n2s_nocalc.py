#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:07:38 2018

@author: cherry
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model
import pandas as pd
import string
from string import digits
import matplotlib.pyplot as plt
from numpy import zeros
import sys
from keras.utils import plot_model
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

embedding_size = 100
units=100
embedding_filename='33_100d_online.emb'
batch_size =64
epochs=50

results_file='results_nocalc_100d_adadelta.txt'
filename='33_nocalc_100d_adadelta.h5'

loss_file='33_nocalc_loss_100d_adadelta.png'
acc_file='33_nocalc_acc_100d_adadelta.png'

lines = pd.read_table('33_absdataset.csv', names=['subject', 'object'])
lines = lines[0:7488]

lines.object = lines.object.apply(lambda x: 'START_ ' + x + ' _END')
lines=shuffle(lines)




all_subject_vocab = set()
for words in lines.subject:
    for word in words.split():
        if word not in all_subject_vocab:
            all_subject_vocab.add(word)

all_object_vocab = set()
for words in lines.object:
    for word in words.split():
        if word not in all_object_vocab:
            all_object_vocab.add(word)

print("no. of words in subjects: ", len(all_object_vocab))
length_list = []
for l in lines.object:
    length_list.append(len(l.split(' ')))
obj_max_length = np.max(length_list)
print('length of largest object', np.max(length_list))

length_list = []
for l in lines.subject:
    length_list.append(len(l.split(' ')))
subject_max_length = np.max(length_list)
print('length of largest subject', np.max(length_list))

input_words = sorted(list(all_subject_vocab))
target_words = sorted(list(all_object_vocab))
num_encoder_tokens = len(all_subject_vocab)
num_decoder_tokens = len(all_object_vocab)



input_token_index = dict([(word, i) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i) for i, word in enumerate(target_words)])


encoder_input_data = np.zeros((len(lines.subject), subject_max_length), dtype='float32')
decoder_input_data = np.zeros((len(lines.object), obj_max_length),dtype='float32')
decoder_target_data = np.zeros((len(lines.object), obj_max_length, num_decoder_tokens),dtype='float32')

for i, (input_text, target_text) in enumerate(zip(lines.subject, lines.object)):
    for t, word in enumerate(input_text.split('\t')):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.


num_lines=lines.shape


em = np.zeros((num_lines[0], units),dtype='float32')

df=pd.read_csv(embedding_filename,sep='\t',header=None)

for i in range(num_lines[0]):
    x = df.index[df.ix[:, 0].str.match(lines.subject[i])]
    em[i] = np.array([[float(digit) for digit in df.ix[x[0], 1].split()]])



decoder_inputs = Input(shape=(None,))
dex = Embedding(num_decoder_tokens, embedding_size)
final_dex = dex(decoder_inputs)
decoder_lstm = LSTM(embedding_size, return_sequences=True, return_state=True)

hidden_states = K.variable(value=em)
cell_states=K.variable(value=np.random.normal(size=(num_encoder_tokens,units)))

decoder_outputs, decoder_h, decoder_c = decoder_lstm(final_dex)
decoder_hc = [decoder_h, decoder_c]
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model(decoder_inputs, decoder_outputs)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

plot_model(model, to_file='node2sequence_t_d.png', show_shapes=True)

model.layers[2].states[0] = hidden_states
model.layers[2].states[1] =cell_states

checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
history=model.fit(decoder_input_data[:6703,:], decoder_target_data[:6703,:,:],batch_size=batch_size, epochs=epochs,validation_split=0.1, verbose=2,callbacks=[checkpoint])

# Inference decoder model

decoder_state_input_h = Input(shape=(None,))
decoder_state_input_c = Input(shape=(None,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]



final_dex2 = dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)


decoder_model = Model([decoder_inputs]+ decoder_states_inputs,[decoder_outputs2] + decoder_states2)

plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


# Function to generate sequences
def decode_sequence(input_seq,seq_index):
    target_seq = np.zeros((1, 1))
    hidden_states2 = em[int(input_seq),:]
    hidden_states2=hidden_states2.reshape(1,units)
    cell_states2 = np.random.randn(units).reshape(1,units)

    states_value = [hidden_states2,cell_states2]

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
                len(decoded_sentence) > 70):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence



with open(results_file,'w')as w:
    for seq_index in range(6703,num_lines[0]):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq,seq_index)
        w.write(str(lines.subject[seq_index: seq_index + 1])+'\t'+decoded_sentence)




plt1.plot(history.history['acc'])
plt1.plot(history.history['val_acc'])
plt1.title('Model accuracy')
plt1.ylabel('Accuracy')
plt1.xlabel('Epoch')
plt1.legend(['Train', 'Test'], loc='upper left')
plt1.savefig(acc_file)
plt1.close()

plt2.plot(history.history['loss'])
plt2.plot(history.history['val_loss'])
plt2.title('Model loss')
plt2.ylabel('Loss')
plt2.xlabel('Epoch')
plt2.legend(['Train', 'Test'], loc='upper left')
plt2.savefig(loss_file)








