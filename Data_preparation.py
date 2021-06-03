#Import Libraries
import json
from types import TracebackType 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

with open('intents.json') as file:
    data = json.load(file)

#print(data)

training_sentences = []  #to collect patterns
training_labels = []    #to collect tags
labels = []             #o collect all tags together
responses = []          #to collect responces

#reading the JSON file and process the required files

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:  # to sortlist unique tags
        labels.append(intent['tag'])

'''
print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n")
print(training_sentences)
print('********************************************\n')
print(training_labels)
print('++++++++++++++++++++++++++++++++++++++++++++\n')
print(labels)
print("---------------------------------\n")
print(responses)

'''   
num_classes = len(labels)

#the label encoder method provided by the Scikit-Learn library in Python

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

#print(training_labels)


#Tokenization

vocab_size = 1000
embedding_dim = 16
'''
Keras offers an Embedding layer that can be used for neural networks on text data.
It requires that the input data be integer encoded, so that each word is represented by a unique integer.
'''

max_len = 20
oov_token = "<OOV>" 
'''
if given, it will be added to word_index and used to replace 
out-of-vocabulary words during text_to_sequence calls
'''

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)  #tokenizer object creation
#print(tokenizer)  #it will give the address of object

tokenizer.fit_on_texts(training_sentences)
#print(tokenizer)  #it will give the address of object

word_index = tokenizer.word_index
#print(word_index) #it will index the words and forming a dictionary

sequences = tokenizer.texts_to_sequences(training_sentences)  #Transforms each text in texts to a sequence(nested list) of integers.
#print(sequences)

padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)  #makes the data in form of array
#print(padded_sequences)



#****************TRAINING A NEURAL NETWORK***************

model = Sequential()  #Sequential groups a linear stack of layers into a tf.keras.Model.
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


model.summary()
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)


#Saving the neural network
model.save("chat_model")


# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
