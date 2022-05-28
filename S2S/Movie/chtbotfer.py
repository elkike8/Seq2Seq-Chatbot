import numpy as np
import pandas as pd
import string
import pickle
import operator
import matplotlib.pyplot as plt
import codecs
import gc

# https://www.kaggle.com/kikegonzalez/seq-to-seq-model-chatbot/edit

# read 
with codecs.open("C:/Users/FerGo/OneDrive/ACIT/2021/Computational Intelligence/Project/S2S/Movie/movie_lines.tsv","rb",encoding="utf-8",errors="ignore") as f:
    lines=f.read().split('\n')

# separate by lines
conversations=[]
for line in lines:
    data=line.split('\t')
    conversations.append(data)

# create a dictionary where each speach by 1 person is indexed with the unique identifier from the base
chats = {}
for tokens in conversations:
    if len(tokens) > 4:
        idx_L=tokens[0].find('L')
        if idx_L !=-1:
            idx=tokens[0][idx_L+1:]
            chat = tokens[4]
            chat=chat[:-2]
            chats[int(idx)] = chat
            
sorted_chats=sorted(chats.items(),key=lambda x:x[0])

# creating a dictionary that groups conversations by using the identifier's coorrelative number

conves_dict = {}
counter = 1
conves_ids = []
for i in range(1, len(sorted_chats)+1):
    if i < len(sorted_chats):
        if (sorted_chats[i][0] - sorted_chats[i-1][0]) == 1:
            # 1つ前の会話の頭の文字がないのを確認
            if sorted_chats[i-1][1] not in conves_ids:
                conves_ids.append(sorted_chats[i-1][1])
            conves_ids.append(sorted_chats[i][1])
        elif (sorted_chats[i][0] - sorted_chats[i-1][0]) > 1:            
            conves_dict[counter] = conves_ids
            conves_ids = []
        counter += 1
    else:
        pass

# reducing every conversation to prompt and answer      

context_and_target=[]
for conves in conves_dict.values():
    if len(conves) % 2 != 0:
        conves = conves[:-1]
    for i in range(0, len(conves), 2):
        context_and_target.append((conves[i], conves[i+1]))
 
# separating the previous into two different lists
        
context, target = zip(*context_and_target)
context = list(context)
target = list(target)
        
# text cleaner

import re
def clean_text(text):    

    text = text.lower()    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

# cleaning the lists
tidy_target = []
for conve in target:
    text = clean_text(conve)
    tidy_target.append(text)

tidy_context = []
for conve in context:
    text = clean_text(conve)
    tidy_context.append(text)
    
# defining variables for the s2s process

bos = "<BOS> "
eos = " <EOS>"
final_target = [bos + conve + eos for conve in tidy_target] 
encoder_inputs = tidy_context
decoder_inputs = final_target

# not sure if this actually does anything
encoder_text = []
for line in encoder_inputs:
    data = line.split("\n")[0]
    encoder_text.append(data)
    
decoder_text = []
for line in decoder_inputs:
    data = line.split("\n")[0]
    decoder_text.append(data)

# list of all promts followed by all answers 
full_text=encoder_text+decoder_text

# =============================================================================
#  model
# =============================================================================

from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 15000
tokenizer = Tokenizer(num_words=VOCAB_SIZE,oov_token='<OOV>')

# fitting

tokenizer.fit_on_texts(full_text)

# creating a dictionary with the ranking of the occurance of words

word_index = tokenizer.word_index
print(len(word_index))
word_index[bos]=len(word_index)+1
word_index[eos]=len(word_index)+1

# creating a dictionary the size of vocab size indexed by occurrance (highest occ = lowest numb)

index2word = {}
for k, v in word_index.items():
    if v < VOCAB_SIZE:
        index2word[v] = k
    if v > VOCAB_SIZE:
        continue

# creating a word index dict the size of the vocab

word2index = {}
for k, v in index2word.items():
    word2index[v] = k
    
# creating sentences with numbers (tokens)

encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

# checking everithing inside vocab size

for seqs in encoder_sequences:
    for seq in seqs:
        if seq > VOCAB_SIZE:
            print(seq)
            break
# updating the vocab size (not really)

VOCAB_SIZE = len(index2word) + 1
VOCAB_SIZE

# transforming the enc/dec to arrays, filling the unused columns with 0

MAX_LEN = 20
from keras.preprocessing.sequence import pad_sequences
encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

# creating array for decoder ourput

num_samples = len(encoder_sequences)
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")


