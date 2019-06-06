"""Script to load the target sentences and process, save them
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
import os
import io
import time
import argparse
import pickle

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

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


def create_dataset(src_path, tgt_path, num_examples):
    src_lines = io.open(src_path, encoding='UTF-8').read().strip().split('\n')
    tgt_lines = io.open(tgt_path, encoding='UTF-8').read().strip().split('\n')

    src_lines = [preprocess_sentence(w) for w in src_lines]
    tgt_lines = [preprocess_sentence(w) for w in tgt_lines]

    return (src_lines, tgt_lines)

def create_gat_dataset(tgt_path, num_examples):
    tgt_lines = io.open(tgt_path, encoding='UTF-8').read().strip().split('\n')
    tgt_lines = [preprocess_sentence(w) for w in tgt_lines]

    return tgt_lines  


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


def load_dataset(src_path, tgt_path, num_examples=None):
    # creating cleaned input, output pairs
    inp_lang, targ_lang = create_dataset(src_path, tgt_path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def load_gat_dataset(adj_path, nodes_path, edges_path, tgt_path, num_examples=None):
    targ_lang = create_gat_dataset(tgt_path, num_examples)
    targ_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    graph_adj = np.load(adj_path) 

    with open(nodes_path, 'rb') as f:
        graph_nodes = pickle.load(f)

    with open(edges_path, 'rb') as edge_f:
        graph_edges = pickle.load(edge_f) 
    
    nodes_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='') 
    nodes_tokenizer.fit_on_texts(graph_nodes) 
    node_tensor = nodes_tokenizer.texts_to_sequences(graph_nodes)
    node_tensor = tf.keras.preprocessing.sequence.pad_sequences(node_tensor,padding='post') 
    
    edges_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='') 
    edges_tokenizer.fit_on_texts(graph_edges) 
    edge_tensor = nodes_tokenizer.texts_to_sequences(graph_edges)
    edge_tensor = tf.keras.preprocessing.sequence.pad_sequences(edge_tensor,padding='post')

    return (graph_adj, node_tensor, nodes_tokenizer, edge_tensor,
            edges_tokenizer, targ_tensor, targ_lang_tokenizer, max_length(targ_tensor))


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


def get_dataset(args):

    input_tensor, target_tensor, input_lang, target_lang = load_dataset(args.src_path, args.tgt_path, args.num_examples)

    BUFFER_SIZE = len(input_tensor)
    BATCH_SIZE = args.batch_size
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    vocab_inp_size = len(input_lang.word_index) + 1
    vocab_tgt_size = len(target_lang.word_index) + 1

    print(len(input_tensor))
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    return dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, vocab_inp_size, vocab_tgt_size

def get_gat_dataset(args):

    (graph_adj, node_tensor, nodes_lang, edge_tensor, edges_lang,
    target_tensor, target_lang, max_length_targ )= load_gat_dataset(args.graph_adj, args.graph_nodes,
                                                    args.graph_edges, args.tgt_path, args.num_examples)

    # Pad the node tensors tp 16 size 
    paddings = tf.constant([[0, 0], [0, 8]])
    node_tensor = tf.pad(node_tensor, paddings)

    BUFFER_SIZE = len(target_tensor)
    BATCH_SIZE = args.batch_size
    steps_per_epoch = len(target_tensor) // BATCH_SIZE
    vocab_tgt_size = len(target_lang.word_index) + 1
    vocab_nodes_size = len(nodes_lang.word_index) + 1
    vocab_edge_size = len(edges_lang.word_index) + 1 

    dataset = tf.data.Dataset.from_tensor_slices((graph_adj, node_tensor, 
                                                    edge_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    return (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
            vocab_tgt_size, vocab_nodes_size, target_lang, max_length_targ)










