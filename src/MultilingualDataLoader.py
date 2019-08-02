from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import pickle
import sentencepiece as spm
from pathlib import Path
import numpy as np
import os
import io
from src.utils.model_utils import convert, max_length, PreProcessSentence
from src.utils.MultilingualUtils import PreProcess
from src.utils.MultilingualUtils import Padding as padding
from src.arguments import get_args

languages = ['eng', 'rus', 'ger']

def LoadMultlingualDataset(args):
    dataset = {}
    CUR_DIR = os.getcwd()
    levels_up = 0
    DATA_PATH = (os.path.normpath(os.path.join(*([CUR_DIR] + [".."] * levels_up)))) + '/data/processed_data/'

    #create vocabs for the source
    src_vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
    target_str = ''

    for lang in languages:

        (dataset[lang + '_train_nodes'], dataset[lang + '_train_labels'],
         dataset[lang + '_train_node1'], dataset[lang + '_train_node2']) = PreProcess(DATA_PATH + lang + '/train_src',
                                                                                      lang)
        (dataset[lang + '_eval_nodes'], dataset[lang + '_eval_labels'],
         dataset[lang + '_eval_node1'], dataset[lang + '_eval_node2']) = PreProcess(DATA_PATH + lang + '/eval_src',
                                                                                    lang)
        (dataset[lang + '_test_nodes'], dataset[lang + '_test_labels'],
         dataset[lang + '_test_node1'], dataset[lang + '_test_node2']) = PreProcess(DATA_PATH + lang + '/test_src',
                                                                                    lang)
        train_tgt = io.open(DATA_PATH + lang + '/train_tgt', encoding='UTF-8').read().strip().split('\n')
        dataset[lang +'_train_tgt'] = [(PreProcessSentence(w, lang)) for w in train_tgt]
        eval_tgt = io.open(DATA_PATH + lang + '/eval_tgt', encoding='UTF-8').read().strip().split('\n')
        dataset[lang +'_eval_tgt'] = [(PreProcessSentence(w, lang)) for w in eval_tgt]
        target_str += (DATA_PATH + lang + '/train_tgt') + ','
        target_str += (DATA_PATH + lang + '/eval_tgt') + ','

        #fit the vocab
        src_vocab.fit_on_texts(dataset[lang+'_train_nodes'])
        src_vocab.fit_on_texts(dataset[lang+'_train_labels'])
        src_vocab.fit_on_texts(dataset[lang+'_train_node1'])
        src_vocab.fit_on_texts(dataset[lang+'_train_node2'])

    os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
    spm.SentencePieceTrainer.Train('--input=' +target_str +' \
                                        --model_prefix=vocabs/' + args.model + '/' + args.lang + '/train_tgt \
                                        --vocab_size=10000 --character_coverage=1.0 --model_type=bpe')
    sp = spm.SentencePieceProcessor()
    sp.load('vocabs/' + args.model + '/' + args.lang + '/train_tgt.model')
    print('Sentencepiece vocab size {}'.format(sp.get_piece_size()))

    # save the vocab file
    os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
    with open(('vocabs/gat/' + args.lang + '/' + args.opt + '_src_vocab'), 'wb+') as fp:
        pickle.dump(src_vocab, fp)

    print('Vocab file saved !\n')

    return dataset, src_vocab, sp

def ProcessMultilingualDataset(args):
    multi_dataset_comps = {}
    dataset, src_vocab, tgt_vocab = LoadMultlingualDataset(args)
    (multi_dataset_comps['nodes'], multi_dataset_comps['labels'],
     multi_dataset_comps['node1'], multi_dataset_comps['node2']) = [], [], [], []

    TRAIN_BUFFER_SIZE = 0
    EVAL_BUFFER_SIZE = 0
    for lang in languages:
        dataset[lang + '_train_tgt'] = [tgt_vocab.encode_as_ids(w) for w in dataset[lang + '_train_tgt']]
        dataset[lang + '_eval_tgt'] = [tgt_vocab.encode_as_ids(w) for w in dataset[lang + '_eval_tgt']]
        dataset[lang + '_train_tgt'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang+'_train_tgt'], padding='post')
        dataset[lang + '_eval_tgt'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang+'_eval_tgt'], padding='post')

        for part in ['train', 'eval', 'test']:
            dataset[lang + '_'+part+'_nodes'] = src_vocab.texts_to_sequences(dataset[lang + '_'+part+'_nodes'])
            dataset[lang + '_'+part+'_labels'] = src_vocab.texts_to_sequences(dataset[lang + '_'+part+'_labels'])
            dataset[lang + '_'+part+'_node1'] = src_vocab.texts_to_sequences(dataset[lang + '_'+part+'_node1'])
            dataset[lang + '_'+part+'_node2'] = src_vocab.texts_to_sequences(dataset[lang + '_'+part+'_node2'])

            dataset[lang + '_'+part+'_nodes'] = padding(
                tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_'+part+'_nodes'],
                                                            padding='post'), 16)
            dataset[lang + '_'+part+'_labels'] = padding(
                tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_'+part+'_labels'],
                                                            padding='post'), 16)
            dataset[lang + '_'+part+'_node1'] = padding(
                tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_'+part+'_node1'],
                                                            padding='post'), 16)
            dataset[lang + '_'+part+'_node2'] = padding(
                tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_'+part+'_node2'],
                                                            padding='post'), 16)

        TRAIN_BUFFER_SIZE += (dataset[lang + '_train_nodes']).shape[0]
        EVAL_BUFFER_SIZE += (dataset[lang + '_eval_nodes']).shape[0]

    MaxSeqSize = max(dataset['eng_train_tgt'].shape[1],
                     dataset['ger_train_tgt'].shape[1],
                     dataset['rus_train_tgt'].shape[1],)

    for lang in languages:
        dataset[lang + '_train_tgt'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_tgt'],
                                                          padding='post'), MaxSeqSize)
        dataset[lang + '_eval_tgt'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_tgt'],
                                                          padding='post'), MaxSeqSize)

    multilingual_target = tf.concat([dataset[lang + '_train_tgt'] for lang in languages], axis=0)
    multilingual_nodes = tf.concat([dataset[lang + '_train_nodes'] for lang in languages], axis=0)
    multilingual_labels = tf.concat([dataset[lang + '_train_labels'] for lang in languages], axis=0)
    multilingual_node1 = tf.concat([dataset[lang + '_train_node1'] for lang in languages], axis=0)
    multilingual_node2 = tf.concat([dataset[lang + '_train_node2'] for lang in languages], axis=0)

    eval_target = tf.concat([dataset[lang + '_eval_tgt'] for lang in languages], axis=0)
    eval_nodes = tf.concat([dataset[lang + '_eval_nodes'] for lang in languages], axis=0)
    eval_labels = tf.concat([dataset[lang + '_eval_labels'] for lang in languages], axis=0)
    eval_node1 = tf.concat([dataset[lang + '_eval_node1'] for lang in languages], axis=0)
    eval_node2 = tf.concat([dataset[lang + '_eval_node2'] for lang in languages], axis=0)

    test_nodes = tf.concat([dataset[lang + '_test_nodes'] for lang in languages], axis=0)
    test_labels = tf.concat([dataset[lang + '_test_labels'] for lang in languages], axis=0)
    test_node1 = tf.concat([dataset[lang + '_test_node1'] for lang in languages], axis=0)
    test_node2 = tf.concat([dataset[lang + '_test_node2'] for lang in languages], axis=0)
    
    BATCH_SIZE = 32
    steps_per_epoch = TRAIN_BUFFER_SIZE // BATCH_SIZE
    src_vocab_size = len(src_vocab.word_index) + 1
    tgt_vocab_size = tgt_vocab.get_piece_size()
    dataset_size = multilingual_target.shape[0]

    multilingual_dataset = tf.data.Dataset.from_tensor_slices((multilingual_nodes, multilingual_labels,
                                                               multilingual_node1, multilingual_node2,
                                                               multilingual_target)).shuffle(TRAIN_BUFFER_SIZE)
    multilingual_dataset = multilingual_dataset.batch(BATCH_SIZE, drop_remainder=True)

    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_nodes, eval_labels,
                                                       eval_node1, eval_node2,
                                                       eval_target)).shuffle(EVAL_BUFFER_SIZE)
    eval_dataset = eval_dataset.batch(BATCH_SIZE, drop_remainder=True)

    test_set = tf.data.Dataset.from_tensor_slices((test_nodes, test_labels,
                                                       test_node1, test_node2))
    test_dataset = test_set.batch(BATCH_SIZE, drop_remainder=True)

    return (multilingual_dataset, eval_dataset, test_dataset, TRAIN_BUFFER_SIZE, EVAL_BUFFER_SIZE, BATCH_SIZE,
            steps_per_epoch, src_vocab, src_vocab_size, tgt_vocab, tgt_vocab_size, MaxSeqSize, dataset_size)