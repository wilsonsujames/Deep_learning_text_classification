import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba as jb


loaded_model = tf.keras.models.load_model('./ReviseModel')

with open('ReviseTokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

a=(lambda x: " ".join([w for w in (jb.cut(x)) ]))("修正電話")
print(a)

# 斷詞很重要

abc=tokenizer.texts_to_sequences([a])
print(abc)
cba = pad_sequences(abc,maxlen=7)
print(cba)
ans=loaded_model.predict(cba)

np.set_printoptions(suppress=True)

print(ans)

# 日期 1
# 電話 2
# 時間 2
# 人數 3