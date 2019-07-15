"""Script to load the target sentences and process, save them
as tf.data files
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import unicodedata
import re
import numpy as np
import os
import io
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


def load_dataset(src_path, tgt_path, lang, num_examples=None):
    # creating cleaned input, output pairs
    inp_lang, targ_lang = create_dataset(src_path, tgt_path, num_examples)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    tokenizer.fit_on_texts(inp_lang)
    tokenizer.fit_on_texts(targ_lang)
    input_tensor = tokenizer.texts_to_sequences(inp_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 padding='post')
    target_tensor = tokenizer.texts_to_sequences(targ_lang)
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                 padding='post')

    os.makedirs(('vocabs/seq2seq/'+lang), exist_ok=True)
    with open(('vocabs/seq2seq/'+lang+'/vocab'), 'wb+') as fp:
        pickle.dump(tokenizer, fp)

    return input_tensor, target_tensor, tokenizer

def load_gat_dataset(nodes_path, labels_path, node1_path, node2_path, tgt_path, lang, num_examples=None):
    targ_lang = create_gat_dataset(tgt_path, num_examples)
    targ_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    with open(nodes_path, 'rb') as f:
        graph_nodes = pickle.load(f)

    with open(labels_path, 'rb') as edge_f:
        edge_labels = pickle.load(edge_f)

    with open(node1_path, 'rb') as role_f:
        node1 = pickle.load(role_f)

    with open(node2_path, 'rb') as role_f:
        node2 = pickle.load(role_f)


    src_vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
    src_vocab.fit_on_texts(graph_nodes)
    src_vocab.fit_on_texts(edge_labels)
    src_vocab.fit_on_texts(node1)
    src_vocab.fit_on_texts(node2)

    node_tensor = src_vocab.texts_to_sequences(graph_nodes)
    node_tensor = tf.keras.preprocessing.sequence.pad_sequences(node_tensor,padding='post') 

    label_tensor = src_vocab.texts_to_sequences(edge_labels)
    label_tensor = tf.keras.preprocessing.sequence.pad_sequences(label_tensor,padding='post')

    node1_tensor = src_vocab.texts_to_sequences(node1)
    node1_tensor = tf.keras.preprocessing.sequence.pad_sequences(node1_tensor,padding='post')

    node2_tensor = src_vocab.texts_to_sequences(node2)
    node2_tensor = tf.keras.preprocessing.sequence.pad_sequences(node2_tensor, padding='post')
    # save all vocabularies
    os.makedirs(('vocabs/gat/'+lang), exist_ok=True)
    with open(('vocabs/gat/'+lang+'/target_vocab'), 'wb+') as fp:
        pickle.dump(targ_lang_tokenizer, fp)
    with open(('vocabs/gat/'+lang+'/src_vocab'), 'wb+') as fp:
        pickle.dump(src_vocab, fp)

    return (node_tensor, label_tensor, node1_tensor, node2_tensor,
            targ_tensor, src_vocab, targ_lang_tokenizer, max_length(targ_tensor))


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


def get_dataset(args):

    input_tensor, target_tensor, lang = load_dataset(args.src_path, args.tgt_path, args.lang, args.num_examples)

    BUFFER_SIZE = len(input_tensor)
    BATCH_SIZE = args.batch_size
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    vocab_size = len(lang.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    return (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
           vocab_size, lang)

def get_gat_dataset(args, lang):

    (node_tensor, label_tensor, node1_tensor, node2_tensor,
     target_tensor, src_vocab, tgt_vocab, max_length_targ) = load_gat_dataset(args.graph_nodes, args.edge_labels,args.edge_node1,
                                                                              args.edge_node2, args.tgt_path, lang)

    # Pad the edge tensor to 16 size
    node_paddings = tf.constant([[0, 0], [0, 1]])
    label_tensor = tf.pad(label_tensor, node_paddings, mode='CONSTANT')
    node1_tensor = tf.pad(node1_tensor, node_paddings, mode='CONSTANT')
    node2_tensor = tf.pad(node2_tensor, node_paddings, mode='CONSTANT')

    BUFFER_SIZE = len(target_tensor)
    BATCH_SIZE = args.batch_size
    steps_per_epoch = len(target_tensor) // BATCH_SIZE
    vocab_tgt_size = len(tgt_vocab.word_index) + 1
    vocab_src_size = len(src_vocab.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((node_tensor, label_tensor,
                                                    node1_tensor, node2_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    return (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, vocab_tgt_size,
            vocab_src_size, src_vocab, tgt_vocab, max_length_targ)