

import tensorflow as tf


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow_addons as tfa

def sequence_to_text(list_of_indices):
  
    with open('tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

def test_model(image_path,question):

  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (224, 224))
  img = tf.keras.applications.inception_v3.preprocess_input(img)


  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

  train_question_seqs = tokenizer.texts_to_sequences(question)
  question_vector = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post',maxlen=22)
  
#add model loaction in load_model
  predictor = tf.keras.models.load_model("model.h5")
  ans = predictor.predict([np.expand_dims(img, axis = 0), np.expand_dims(question_vector.reshape(22,1),axis = 0)])
  ans = np.int8(ans == ans.max())
  
#add file loaction
  object_categories = np.load("encoder.npy")
  onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, categories=object_categories)
  gg = onehot_encoder.fit_transform(np.array(1).reshape(-1, 1))
  ans = onehot_encoder.inverse_transform(ans)


#add file loaction in mp.load()
  label_encoder = LabelEncoder()
  label_encoder.classes_ = np.load("classes.npy")
  ans = label_encoder.inverse_transform(ans.ravel())
  question = list(map(sequence_to_text, question_vector))

  return ans
# question = ['<start> where is person sitting? <end>']