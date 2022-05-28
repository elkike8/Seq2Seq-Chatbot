import pandas as pd
import numpy as np
import re

# https://www.kaggle.com/currie32/a-south-park-chatbot
# https://www.youtube.com/watch?v=DItR-l59i6M&list=PLTuKYqpidPXbulRHl8HL7JLRQXwDlqpLO&index=5

southpark = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Computational Intelligence/Project/S2S South Park/All-seasons.csv")[:1000]

# define function to clean the text 

def clean_text(text):

    text = text.lower()
    
    text = re.sub(r"\n", "",  text)
    text = re.sub(r"[-()]", "", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\,", " , ", text)
    text = re.sub(r"\"", " \" ", text)
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

# Clean the scripts and add them to the same list.
text = []

for line in southpark.Line:
    text.append(clean_text(line))
    
# Find the length of lines in number of words + Characters
lengths = []
for line in text:
    lengths.append(len(line.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


# Using lines shorter than the 95 percentile
max_line_length = int(np.percentile(lengths, 95) - np.percentile(lengths, 95)%10)

short_text = []
for line in text:
    if len(line.split()) <= max_line_length:
        short_text.append(line)


# Create a dictionary for the frequency of the vocabulary
vocab = {}
for line in short_text:
    for word in line.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

# Limit the vocabulary to words used more than 3 times.
threshold = 3
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1
print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)

# create dictionaries to provide a unique integer for each word.

vocab_to_int = {}
word_num = 1
for k,v in vocab.items():
    if v >= threshold:
        vocab_to_int[k] = word_num
        word_num += 1
                
# Add the unique tokens to the vocabulary dictionaries.
codes = ['<PAD>','<EOS>','<UNK>','<BOS>']

for code in codes:
    vocab_to_int[code] = len(vocab_to_int)+1
    
    
# have <PAD> as 0
vocab_to_int["<PAD>"] = 0

# inv answers dict 
inv_vocab = {w:v for v, w in vocab_to_int.items()}

# deleting extra variables
del (line, text, code, codes, count, k, threshold, v, vocab, word_num, word)

# Create the questions and answers texts.
# The answer text is the line following the question text.
q_text = short_text[:-1]
a_text = short_text[1:]

# adding tokens to sentences
for i in range(len(a_text)):
    a_text[i] = "<BOS>" + a_text[i] + "<EOS>"
    
# Convert the text to integers. 
# Replace any words that are not in the respective vocabulary with <UNK> (unknown)
q_int = []
for line in q_text:
    sentence = []
    for word in line.split():
        if word not in vocab_to_int:
            sentence.append(vocab_to_int['<UNK>'])
        else:
            sentence.append(vocab_to_int[word])
    q_int.append(sentence)
    
a_int = []
for line in a_text:
    sentence = []
    for word in line.split():
        if word not in vocab_to_int:
            sentence.append(vocab_to_int['<UNK>'])
        else:
            sentence.append(vocab_to_int[word])
    a_int.append(sentence)

del(a_text, i, line, q_text, sentence, short_text, word)

# =============================================================================
#  modeling
# =============================================================================

# creating arrays to feed the model

from tensorflow.keras.preprocessing.sequence import pad_sequences
q_int = pad_sequences(q_int, max_line_length, padding='post', truncating='post')
a_int = pad_sequences(a_int, max_line_length, padding='post', truncating='post')

# creating the -1 array

decoder_output = []
for i in a_int:
    decoder_output.append(i[1:]) 
decoder_output = pad_sequences(decoder_output, max_line_length, padding='post', truncating='post')
del (i)

# making a 3D matrix of the output which is expected by lstm

from tensorflow.keras.utils import to_categorical
decoder_output = to_categorical(decoder_output, len(vocab_to_int))

# defining the model's layers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input


enc_inp = Input(shape=(max_line_length, ))
dec_inp = Input(shape=(max_line_length, ))


VOCAB_SIZE = len(vocab_to_int)

#creating the embeding layer to reduce the dimentionality (the funnel)

embed = Embedding(VOCAB_SIZE+1, output_dim=50, 
                  input_length=max_line_length,
                  trainable=True                  
                  )
enc_embed = embed(enc_inp)

# creating the lstm layers

enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_out, h, c = enc_lstm(enc_embed)
enc_states = [h, c]


dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
dec_out, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

# creating the dense output layer

dense = Dense(VOCAB_SIZE, activation='softmax')

dense_out = dense(dec_out)

# assembling and fitting the model

model = Model([enc_inp, dec_inp], dense_out)

model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')

model.fit([q_int, a_int],decoder_output,epochs=15)


# =============================================================================
#  Creating an inference model
# =============================================================================

# create model using previous previous model states and input layer

enc_model = Model([enc_inp], enc_states)


# decoder Model 
# input layers
decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# getting outputs and states for the model from the previous trained model
decoder_outputs, state_h, state_c = dec_lstm(dec_embed , 
                                    initial_state=decoder_states_inputs)


decoder_states = [state_h, state_c]

# defining the decoder model
dec_model = Model([dec_inp]+ decoder_states_inputs,
                                      [decoder_outputs]+ decoder_states)


model.summary()

enc_model.summary()
dec_model.summary()

print("##########################################")
print("#       start chatting ver. 1.0          #")
print("##########################################")

chatting = ""
while chatting != "take me out of the matrix":
    
    chatting  = input("you : ")
 
    chatting = clean_text(chatting)

    chatting = [chatting]

    txt = []
    
    for x in chatting:        
        lst = []
        for y in x.split():
            try:
                lst.append(vocab_to_int[y])
            except:
                lst.append(vocab_to_int['<UNK>'])
        txt.append(lst)

    txt = pad_sequences(txt, max_line_length, padding='post')

    stat = enc_model.predict( txt )
    
    empty_target_seq = np.zeros( ( 1 , 1) )

    empty_target_seq[0, 0] = vocab_to_int["<BOS>"]

    stop_condition = False
    decoded_translation = ""

    while not stop_condition :

        dec_outputs , h, c= dec_model.predict([ empty_target_seq] + stat )
        decoder_concat_input = dense(dec_outputs)


        sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )


        sampled_word = inv_vocab[sampled_word_index] + " "


        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word  

        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > max_line_length:
            stop_condition = True 

        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index

        stat = [h, c]  

    print("chatbot : ", decoded_translation )
    print("==============================================")  
    
