from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import pickle
import argparse
from pathlib import Path
import numpy as np
import os
from src.utils.model_utils import convert, max_length

languages = ['eng', 'rus', 'ger']

def padding(tensor, max_length):
    padding = tf.constant([[0, 0], [0, max_length - tensor.shape[1]]])
    padded_tensor = tf.pad(tensor, padding, mode='CONSTANT')

    return padded_tensor

def load_gat_multidataset(args):
    """

    :param args:
    :type args:
    :return:
    :rtype:
    """
    dataset = {}
    CUR_DIR = os.getcwd()
    levels_up = 0
    DATA_PATH = (os.path.normpath(os.path.join(*([CUR_DIR]+[".."]*levels_up))))+'/data/processed_graphs/'

    TRAIN_DIRS = [DATA_PATH+lang+'/'+args.model+'/'+args.opt+'_train' for lang in languages]
    EVAL_DIRS = [DATA_PATH + lang + '/' + args.model + '/' + args.opt + '_eval' for lang in languages]

    vocab = tf.keras.preprocessing.text.Tokenizer(filters='')

    for (train_dir, eval_dir, lang) in zip(TRAIN_DIRS, EVAL_DIRS, languages):
        with open(train_dir, 'rb') as f:
            train_set = pickle.load(f)
        with open(eval_dir, 'rb') as f:
            eval_set = pickle.load(f)

        train_input, train_tgt = zip(*train_set)
        eval_input, eval_tgt = zip(*eval_set)

        (dataset[lang+'_train_nodes'], dataset[lang+'_train_labels'],
         dataset[lang+'_train_node1'], dataset[lang+'_train_node2']) = zip(*train_input)
        (dataset[lang + '_eval_nodes'], dataset[lang + '_eval_labels'],
         dataset[lang + '_eval_node1'], dataset[lang + '_eval_node2']) = zip(*eval_input)
        dataset[lang + '_train_tgt'] = train_tgt
        dataset[lang + '_eval_tgt'] = eval_tgt

        [x.append(lang) for x in dataset[lang+'_train_nodes']]
        #fit the vocabs on the dataset
        vocab.fit_on_texts(dataset[lang+'_train_nodes'])
        vocab.fit_on_texts(dataset[lang + '_train_labels'])
        vocab.fit_on_texts(dataset[lang + '_train_node1'])
        vocab.fit_on_texts(dataset[lang + '_train_node2'])
        vocab.fit_on_texts(dataset[lang + '_train_tgt'])

    #save the dataset and vocab
    os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
    with open(('vocabs/gat/' + args.lang + '/' + args.opt + '_vocab'), 'wb+') as fp:
        pickle.dump(vocab, fp)

    if args.use_colab is not None:
        from google.colab import drive

        drive.mount('/content/gdrive', force_remount=True)
        OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/' + args.lang + '/' + args.model
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    else:
        OUTPUT_DIR = 'data/processed_graphs/' + args.lang + '/' + args.model
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
    with open(OUTPUT_DIR + '/' + args.opt + '_train', 'wb') as fp:
        pickle.dump(dataset, fp)
    print('Dumped the dataset and Vocab !')

    return dataset, vocab

def process_gat_multidataset(args):
    multi_dataset_comps = {}
    dataset, vocab = load_gat_multidataset(args)
    (multi_dataset_comps['nodes'], multi_dataset_comps['labels'],
    multi_dataset_comps['node1'], multi_dataset_comps['node2']) = [], [], [], []

    TRAIN_BUFFER_SIZE = 0
    EVAL_BUFFER_SIZE = 0
    for lang in languages:
        dataset[lang+'_train_nodes'] = vocab.texts_to_sequences(dataset[lang+'_train_nodes'])
        dataset[lang + '_train_labels'] = vocab.texts_to_sequences(dataset[lang + '_train_labels'])
        dataset[lang + '_train_node1'] = vocab.texts_to_sequences(dataset[lang + '_train_node1'])
        dataset[lang + '_train_node2'] = vocab.texts_to_sequences(dataset[lang + '_train_node2'])

        dataset[lang + '_eval_nodes'] = vocab.texts_to_sequences(dataset[lang + '_eval_nodes'])
        dataset[lang + '_eval_labels'] = vocab.texts_to_sequences(dataset[lang + '_eval_labels'])
        dataset[lang + '_eval_node1'] = vocab.texts_to_sequences(dataset[lang + '_eval_node1'])
        dataset[lang + '_eval_node2'] = vocab.texts_to_sequences(dataset[lang + '_eval_node2'])

        dataset[lang + '_train_tgt'] = vocab.texts_to_sequences(dataset[lang + '_train_tgt'])
        dataset[lang + '_eval_tgt'] = vocab.texts_to_sequences(dataset[lang + '_eval_tgt'])

        dataset[lang + '_train_nodes'] = padding(tf.keras.preprocessing.sequence.pad_sequences(dataset[lang+'_train_nodes'],
                                                                                        padding='post'), 16)
        dataset[lang + '_train_labels'] = padding(tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_labels'],
                                                                                       padding='post'), 16)
        dataset[lang + '_train_node1'] = padding(tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_node1'],
                                                                                       padding='post'), 16)
        dataset[lang + '_train_node2'] = padding(tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_node2'],
                                                                                       padding='post'), 16)
        dataset[lang + '_train_tgt'] = padding(tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_tgt'],
                                                                                       padding='post'), 222)

        dataset[lang + '_eval_nodes'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_nodes'],
                                                          padding='post'), 16)
        dataset[lang + '_eval_labels'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_labels'],
                                                          padding='post'), 16)
        dataset[lang + '_eval_node1'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_node1'],
                                                          padding='post'), 16)
        dataset[lang + '_eval_node2'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_node2'],
                                                          padding='post'), 16)
        dataset[lang + '_eval_tgt'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_tgt'],
                                                          padding='post'), 222)

        TRAIN_BUFFER_SIZE += (dataset[lang+'_train_nodes']).shape[0]
        EVAL_BUFFER_SIZE += (dataset[lang + '_eval_nodes']).shape[0]

    multilingual_target = tf.concat([dataset[lang+'_train_tgt'] for lang in languages], axis=0)
    multilingual_nodes = tf.concat([dataset[lang+'_train_nodes'] for lang in languages], axis=0)
    multilingual_labels = tf.concat([dataset[lang + '_train_labels'] for lang in languages], axis=0)
    multilingual_node1 = tf.concat([dataset[lang + '_train_node1'] for lang in languages], axis=0)
    multilingual_node2 = tf.concat([dataset[lang + '_train_node2'] for lang in languages], axis=0)

    eval_target = tf.concat([dataset[lang + '_eval_tgt'] for lang in languages], axis=0)
    eval_nodes = tf.concat([dataset[lang + '_eval_nodes'] for lang in languages], axis=0)
    eval_labels = tf.concat([dataset[lang + '_eval_labels'] for lang in languages], axis=0)
    eval_node1 = tf.concat([dataset[lang + '_eval_node1'] for lang in languages], axis=0)
    eval_node2 = tf.concat([dataset[lang + '_eval_node2'] for lang in languages], axis=0)

    BATCH_SIZE = 32
    steps_per_epoch = TRAIN_BUFFER_SIZE // BATCH_SIZE
    vocab_size = len(vocab.word_index) + 1
    dataset_size = multilingual_target.shape[0]

    multilingual_dataset = tf.data.Dataset.from_tensor_slices((multilingual_nodes, multilingual_labels,
                                                  multilingual_node1, multilingual_node2, multilingual_target)).shuffle(TRAIN_BUFFER_SIZE)
    multilingual_dataset = multilingual_dataset.batch(BATCH_SIZE, drop_remainder=True)

    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_nodes, eval_labels,
                                                               eval_node1, eval_node2,
                                                               eval_target)).shuffle(EVAL_BUFFER_SIZE)
    eval_dataset = eval_dataset.batch(BATCH_SIZE, drop_remainder=True)

    return (multilingual_dataset, eval_dataset, TRAIN_BUFFER_SIZE, EVAL_BUFFER_SIZE,
            BATCH_SIZE, steps_per_epoch, vocab_size, vocab, multilingual_target.shape[-1], dataset_size)