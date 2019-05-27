"""Script to load the target sentences annd process, save them 
as tf.data files 
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
tf.enable_eager_execution()


import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

import unicodedata
import re 
import numpy as np 
import numpy as np 
import os 
import io 
import time 
import argparse 

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
'--path', type=str, required=True, help='Path to source.lex file')
parser.add_argument(
'--batch_size', type=int, required=True, help='Batch size')
parser.add_argument(
'--emb_dim', type=int, required=True, help='Embedding dimension')
args = parser.parse_args()


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                            padding='post')

    return tensor, lang_tokenizer

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    lines = [preprocess_sentence(w) for w in lines]

    return lines


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang = create_dataset(path, num_examples)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return target_tensor, targ_lang_tokenizer

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))


if __name__ == "__main__":
    path = args.path 
    dest = create_dataset(path, None)
    target_tensor, targ_lang = load_dataset(path, None)
    max_length_target = max_length(target_tensor)
    BUFFER_SIZE = len(target_tensor)
    steps_per_epoch = BUFFER_SIZE//args.batch_size
    embedding_dim = args.emb_dim 
    vocab_tar_size = len(targ_lang.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices(target_tensor).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    

    
