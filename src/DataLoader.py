"""Script to load the target sentences and process, save them
as tf.data files
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pickle

import numpy as np
import sentencepiece as spm
import tensorflow as tf

from src.utils.model_utils import max_length, _tensorize, Padding as padding


def LoadDataset(train_path, eval_path, test_path,
                vocab_path, sentencepiece, num_examples=None):
    # load the train and eval datasets
    with open(train_path, 'rb') as f:
        train_set = pickle.load(f)
    with open(eval_path, 'rb') as f:
        eval_set = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_set = pickle.load(f)

    train_inp, train_tgt = zip(*train_set)
    eval_inp, eval_tgt = zip(*eval_set)

    # load the vocab
    if sentencepiece == 'True':
        sp = spm.SentencePieceProcessor()
        sp.load(vocab_path)

        input_tensor = [sp.encode_as_ids(w) for w in train_inp]
        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                     padding='post')
        target_tensor = [sp.encode_as_ids(w) for w in train_tgt]
        target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                      padding='post')
        eval_inp = [sp.encode_as_ids(w) for w in eval_inp]
        eval_inp = tf.keras.preprocessing.sequence.pad_sequences(eval_inp,
                                                                 padding='post')
        test_inp = [sp.encode_as_ids(w) for w in test_set]
        test_inp = tf.keras.preprocessing.sequence.pad_sequences(test_inp,
                                                                 padding='post')

        return input_tensor, target_tensor, \
               eval_inp, test_inp, sp
    else:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        input_tensor = _tensorize(vocab, train_inp)
        target_tensor = _tensorize(vocab, train_tgt)
        eval_inp = _tensorize(vocab, eval_inp)
        eval_tgt = _tensorize(vocab, eval_tgt)
        test_inp = _tensorize(vocab, test_set)

        return input_tensor, target_tensor, \
               eval_inp, eval_tgt, test_inp, vocab, max_length(target_tensor)


def LoadGatDataset(train_path, eval_path, test_path, srv_vocab,
                   tgt_vocab, opt, sentencepiece, lang, num_examples=None):
    dataset = {}
    if opt == 'reif':
        # load the train and eval datasets
        with open(train_path, 'rb') as f:
            train_set = pickle.load(f)
        with open(eval_path, 'rb') as f:
            eval_set = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_set = pickle.load(f)

        # load vocab
        if sentencepiece == 'True':
            sp = spm.SentencePieceProcessor()
            sp.load(tgt_vocab)
        with open(srv_vocab, 'rb') as f:
            src_vocab = pickle.load(f)

        train_input, train_tgt = zip(*train_set)
        eval_input, eval_tgt = zip(*eval_set)
        (train_nodes, train_labels, train_node1, train_node2) = zip(*train_input)
        (eval_nodes, eval_labels, eval_node1, eval_node2) = zip(*eval_input)
        (test_nodes, test_labels, test_node1, test_node2) = zip(*test_set)

        train_node_tensor = _tensorize(src_vocab, train_nodes)
        train_label_tensor = _tensorize(src_vocab, train_labels)
        train_node1_tensor = _tensorize(src_vocab, train_node1)
        train_node2_tensor = _tensorize(src_vocab, train_node2)

        eval_node_tensor = _tensorize(src_vocab, eval_nodes)
        eval_label_tensor = _tensorize(src_vocab, eval_labels)
        eval_node1_tensor = _tensorize(src_vocab, eval_node1)
        eval_node2_tensor = _tensorize(src_vocab, eval_node2)

        test_node_tensor = _tensorize(src_vocab, test_nodes)
        test_label_tensor = _tensorize(src_vocab, test_labels)
        test_node1_tensor = _tensorize(src_vocab, test_node1)
        test_node2_tensor = _tensorize(src_vocab, test_node1)

        #######exp######
        if sentencepiece == 'True':
            train_tgt_tensor = [sp.encode_as_ids(w) for w in train_tgt]
            train_tgt_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tgt_tensor, padding='post')
            target_vocab = sp
        else:
            train_tgt_tensor = src_vocab.texts_to_sequences(train_tgt)
            train_tgt_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tgt_tensor, padding='post')
            target_vocab = src_vocab

        return (train_node_tensor, train_label_tensor, train_node1_tensor,
                train_node2_tensor, train_tgt_tensor, eval_node_tensor, eval_label_tensor,
                eval_node1_tensor, eval_node2_tensor, test_node_tensor, test_label_tensor,
                test_node1_tensor, test_node2_tensor, src_vocab, target_vocab, max_length(train_tgt_tensor))

    else:
        # load the train and eval datasets
        with open(train_path, 'rb') as f:
            train_set = pickle.load(f)
        with open(eval_path, 'rb') as f:
            eval_set = pickle.load(f)
        # load vocab
        with open(srv_vocab, 'rb') as f:
            src_vocab = pickle.load(f)

        train_input, train_tgt = zip(*train_set)
        eval_input, eval_tgt = zip(*eval_set)
        (train_adj, train_nodes, train_roles, train_edges) = zip(*train_input)
        (eval_adj, eval_nodes, eval_roles, eval_edges) = zip(*eval_input)

        train_node_tensor = _tensorize(src_vocab, train_nodes)
        train_role_tensor = _tensorize(src_vocab, train_roles)
        train_edges_tensor = _tensorize(src_vocab, train_edges)
        train_tgt_tensor = _tensorize(src_vocab, train_tgt)

        eval_node_tensor = _tensorize(src_vocab, eval_nodes)
        eval_role_tensor = _tensorize(src_vocab, eval_roles)
        eval_edges_tensor = _tensorize(src_vocab, eval_edges)

        return (train_adj, train_node_tensor, train_role_tensor, train_edges_tensor, train_tgt_tensor, eval_adj,
                eval_node_tensor, eval_role_tensor, eval_edges_tensor, src_vocab, max_length(train_tgt_tensor))


def GetDataset(args):
    input_tensor, target_tensor, \
    eval_tensor, eval_tgt, test_inp, lang, max_seq_len = LoadDataset(args.train_path, args.eval_path, args.test_path,
                                                                     args.src_vocab, args.sentencepiece,
                                                                     args.num_examples)

    BUFFER_SIZE = len(input_tensor)
    BATCH_SIZE = args.batch_size
    steps_per_epoch = len(input_tensor) // BATCH_SIZE

    if args.sentencepiece == 'True':
        vocab_size = lang.get_piece_size()
    else:
        vocab_size = len(lang.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    eval_set = tf.data.Dataset.from_tensor_slices((eval_tensor, eval_tgt))
    test_set = tf.data.Dataset.from_tensor_slices((test_inp))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    eval_set = eval_set.batch(BATCH_SIZE, drop_remainder=True)
    test_set = test_set.batch(BATCH_SIZE, drop_remainder=True)
    dataset_size = target_tensor.shape[0]

    return (dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
            vocab_size, lang, dataset_size, max_seq_len)


def GetGATDataset(args, set=None):
    if args.opt == 'reif':
        (train_nodes, train_labels, train_node1, train_node2, train_tgt_tensor,
         eval_nodes, eval_labels, eval_node1, eval_node2, test_nodes, test_labels,
         test_node1, test_node2, src_vocab, tgt_vocab, max_length_targ) = LoadGatDataset(args.train_path,
                                                                                         args.eval_path,
                                                                                         args.test_path, args.src_vocab,
                                                                                         args.tgt_vocab, args.opt,
                                                                                         args.sentencepiece, args.lang)

        node_tensor = padding(train_nodes, 16)
        label_tensor = padding(train_labels, 16)
        node1_tensor = padding(train_node1, 16)
        node2_tensor = padding(train_node2, 16)

        eval_nodes = padding(eval_nodes, 16)
        eval_labels = padding(eval_labels, 16)
        eval_node1 = padding(eval_node1, 16)
        eval_node2 = padding(eval_node2, 16)

        test_nodes = padding(test_nodes, 16)
        test_labels = padding(test_labels, 16)
        test_node1 = padding(test_node1, 16)
        test_node2 = padding(test_node2, 16)

        print('\nTrain Tensor shapes (nodes, labels, node1, node2, target) : ')
        print(node_tensor.shape, label_tensor.shape, node1_tensor.shape, node2_tensor.shape, train_tgt_tensor.shape)
        print('\nEval Tensor shapes (nodes, labes, node1, node2) : ')
        print(eval_nodes.shape, eval_labels.shape, eval_node1.shape, eval_node2.shape)
        print('\nTest Tensor shapes (nodes, labes, node1, node2) : ')
        print(test_nodes.shape, test_labels.shape, test_node1.shape, test_node2.shape)

        BUFFER_SIZE = len(train_tgt_tensor)
        BATCH_SIZE = args.batch_size
        steps_per_epoch = len(train_tgt_tensor) // BATCH_SIZE
        src_vocab_size = len(src_vocab.word_index) + 1
        if args.sentencepiece == 'True':
            tgt_vocab_size = tgt_vocab.get_piece_size()
        else:
            tgt_vocab_size = len(tgt_vocab.word_index) + 1

        dataset_size = train_tgt_tensor.shape[0]

        dataset = tf.data.Dataset.from_tensor_slices((node_tensor, label_tensor,
                                                      node1_tensor, node2_tensor, train_tgt_tensor)).shuffle(
            BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        eval_set = tf.data.Dataset.from_tensor_slices((eval_nodes, eval_labels,
                                                       eval_node1, eval_node2))
        eval_set = eval_set.batch(BATCH_SIZE, drop_remainder=True)

        test_set = tf.data.Dataset.from_tensor_slices((test_nodes, test_labels,
                                                       test_node1, test_node2))
        test_set = test_set.batch(BATCH_SIZE, drop_remainder=True)

        if set == None:
            return (dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
                    src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab,
                    max_length_targ, dataset_size)
        elif set == 'test':
            return (test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
                    src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab)

    else:
        (train_adj, train_node_tensor, train_role_tensor,
         train_edges_tensor, train_tgt_tensor, eval_adj, eval_node_tensor,
         eval_role_tensor, eval_edges_tensor, src_vocab, max_length) = LoadGatDataset(args.train_path, args.eval_path,
                                                                                      args.vocab_path, args.opt,
                                                                                      args.lang)

        node_tensor = padding(train_node_tensor, 16)
        role_tensor = padding(train_role_tensor, 16)
        edge_tensor = padding(train_edges_tensor, 16)

        eval_node_tensor = padding(eval_node_tensor, 16)
        eval_role_tensor = padding(eval_role_tensor, 16)
        eval_edge_tensor = padding(eval_edges_tensor, 16)

        print('\nTrain Tensor shapes (nodes, roles, edges, target) : ')
        print(node_tensor.shape, role_tensor.shape, edge_tensor.shape, train_tgt_tensor.shape)
        print('\nEval Tensor shapes (nodes, roles, edges) : ')
        print(eval_node_tensor.shape, eval_role_tensor.shape, eval_edge_tensor.shape)

        BUFFER_SIZE = len(train_tgt_tensor)
        BATCH_SIZE = args.batch_size
        steps_per_epoch = len(train_tgt_tensor) // BATCH_SIZE
        src_vocab_size = len(src_vocab.word_index) + 1
        dataset_size = train_tgt_tensor.shape[0]

        # convert adj list to numpy array
        train_adj = np.array(train_adj)

        dataset = tf.data.Dataset.from_tensor_slices((train_adj, node_tensor,
                                                      role_tensor, edge_tensor, train_tgt_tensor)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        eval_set = tf.data.Dataset.from_tensor_slices((eval_adj, eval_node_tensor,
                                                       eval_role_tensor, eval_edge_tensor))
        eval_set = eval_set.batch(BATCH_SIZE, drop_remainder=True)

        return (dataset, eval_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
                src_vocab_size, src_vocab, max_length, dataset_size)
