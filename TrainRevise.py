import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import pandas as pd
import jieba as jb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

# ans={"unknown":0,"date":1,'time':2,"peoplecount":3,"phone":4}

df = pd.read_csv(r'revise.csv',encoding='utf-8')

df['text'] = df['text'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) ]))

X_train, X_test, Y_train, Y_test = train_test_split(df[['text']],df[["label"]], test_size = 0.1)

tokenizer = Tokenizer( )
tokenizer.fit_on_texts(X_train["text"].values)

with open('ReviseTokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

X_train_sequences = tokenizer.texts_to_sequences(X_train["text"].values)
print(X_train_sequences)
X_train_padded = pad_sequences(X_train_sequences)
print(X_train_padded)

print("等等:::::")
print(X_train_padded[0])
print("***********")

X_test_sequences = tokenizer.texts_to_sequences(X_test["text"].values)
X_test_padded = pad_sequences(X_test_sequences)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

num_epochs = 300
history=model.fit(X_train_padded, 
        Y_train, 
        epochs=num_epochs, 
        validation_data=(X_test_padded, Y_test), 
        verbose=2)


def plot_graphs(history, string):    
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


model.save('./ReviseModel')


























