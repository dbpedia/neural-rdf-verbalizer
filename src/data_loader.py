"""Script to load the target sentences and process, save them
as tf.data files
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import pickle
import numpy as np
import os

def max_length(tensor):
    return max(len(t) for t in tensor)

def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t.numpy()]))

def load_dataset(train_path, eval_path, vocab_path, lang, num_examples=None):
    # load the train and eval datasets
    with open(train_path, 'rb') as f:
        train_set = pickle.load(f)
    with open(eval_path, 'rb') as f:
        eval_set = pickle.load(f)
    # load vocab
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    train_inp, train_tgt = zip(*train_set)
    eval_inp, eval_tgt = zip(*eval_set)

    input_tensor = vocab.texts_to_sequences(train_inp)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 padding='post')
    target_tensor = vocab.texts_to_sequences(train_tgt)
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                 padding='post')
    eval_inp = vocab.texts_to_sequences(eval_inp)
    eval_inp = tf.keras.preprocessing.sequence.pad_sequences(eval_inp,
                                                                 padding='post')

    return input_tensor, target_tensor, \
           eval_inp, vocab

def load_gat_dataset(train_path, eval_path, vocab_path, opt, lang, num_examples=None):
    if opt == 'reif':
        #load the train and eval datasets
        with open(train_path, 'rb') as f:
            train_set = pickle.load(f)
        with open(eval_path, 'rb') as f:
            eval_set = pickle.load(f)

        #load vocab
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)  

        train_input, train_tgt = zip(*train_set)
        eval_input, eval_tgt = zip(*eval_set)
        (train_nodes, train_labels, train_node1, train_node2) = zip(*train_input)
        (eval_nodes, eval_labels, eval_node1, eval_node2) = zip(*eval_input)
        
        train_node_tensor = vocab.texts_to_sequences(train_nodes)
        eval_node_tensor = vocab.texts_to_sequences(eval_nodes)
        train_node_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_node_tensor,padding='post')
        eval_node_tensor = tf.keras.preprocessing.sequence.pad_sequences(eval_node_tensor, padding='post')

        train_label_tensor = vocab.texts_to_sequences(train_labels)
        eval_label_tensor = vocab.texts_to_sequences(eval_labels)
        train_label_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_label_tensor, padding='post')
        eval_label_tensor = tf.keras.preprocessing.sequence.pad_sequences(eval_label_tensor, padding='post')

        train_node1_tensor = vocab.texts_to_sequences(train_node1)
        eval_node1_tensor = vocab.texts_to_sequences(eval_node1)
        train_node1_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_node1_tensor, padding='post')
        eval_node1_tensor = tf.keras.preprocessing.sequence.pad_sequences(eval_node1_tensor, padding='post')

        train_node2_tensor = vocab.texts_to_sequences(train_node2)
        eval_node2_tensor = vocab.texts_to_sequences(eval_node2)
        train_node2_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_node2_tensor, padding='post')
        eval_node2_tensor = tf.keras.preprocessing.sequence.pad_sequences(eval_node2_tensor, padding='post')

        train_tgt_tensor = vocab.texts_to_sequences(train_tgt)
        train_tgt_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tgt_tensor, padding='post')

        return (train_node_tensor, train_label_tensor, train_node1_tensor, train_node2_tensor, train_tgt_tensor,
                eval_node_tensor, eval_label_tensor, eval_node1_tensor, eval_node2_tensor, vocab, max_length(train_tgt_tensor))
    else:
        #load the train and eval datasets
        with open(train_path, 'rb') as f:
            train_set = pickle.load(f)
        with open(eval_path, 'rb') as f:
            eval_set = pickle.load(f)
        #load vocab
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)  

        train_input, train_tgt = zip(*train_set)
        eval_input, eval_tgt = zip(*eval_set)
        (train_adj, train_nodes, train_roles, train_edges) = zip(*train_input)
        (eval_adj, eval_nodes, eval_roles, eval_edges) = zip(*eval_input)

        train_node_tensor = vocab.texts_to_sequences(train_nodes)
        eval_node_tensor = vocab.texts_to_sequences(eval_nodes)
        train_node_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_node_tensor,padding='post')
        eval_node_tensor = tf.keras.preprocessing.sequence.pad_sequences(eval_node_tensor, padding='post')

        train_role_tensor = vocab.texts_to_sequences(train_roles)
        eval_role_tensor = vocab.texts_to_sequences(eval_roles)
        train_role_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_role_tensor, padding='post')
        eval_role_tensor = tf.keras.preprocessing.sequence.pad_sequences(eval_role_tensor, padding='post')

        train_edges_tensor = vocab.texts_to_sequences(train_edges)
        eval_edges_tensor = vocab.texts_to_sequences(eval_edges)
        train_edges_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_edges_tensor, padding='post')
        eval_edges_tensor = tf.keras.preprocessing.sequence.pad_sequences(eval_edges_tensor, padding='post')

        train_tgt_tensor = vocab.texts_to_sequences(train_tgt)
        train_tgt_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tgt_tensor, padding='post')

        return (train_adj, train_node_tensor, train_role_tensor, train_edges_tensor, train_tgt_tensor, eval_adj, 
                eval_node_tensor, eval_role_tensor, eval_edges_tensor, vocab, max_length(train_tgt_tensor))

def get_dataset(args):

    input_tensor, target_tensor, eval_tensor, lang = load_dataset(args.train_path, args.eval_path, args.vocab_path, args.lang, args.num_examples)

    BUFFER_SIZE = len(input_tensor)
    BATCH_SIZE = args.batch_size
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    vocab_size = len(lang.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset_size = target_tensor.shape[0]

    return (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
           vocab_size, lang, dataset_size)

def get_gat_dataset(args):
    if args.opt == 'reif':
        (train_nodes, train_labels, train_node1, train_node2, train_tgt_tensor,
        eval_nodes, eval_labels, eval_node1, eval_node2, vocab, max_length_targ) = load_gat_dataset(args.train_path, args.eval_path, 
                                                                                                    args.vocab_path, args.opt, args.lang)

        node_padding = tf.constant([[0, 0], [0, 16-train_nodes.shape[1]]])
        node_tensor = tf.pad(train_nodes, node_padding, mode='CONSTANT')
        label_padding = tf.constant([[0, 0], [0, 16-train_labels.shape[1]]])
        label_tensor = tf.pad(train_labels, label_padding, mode='CONSTANT')
        node1_paddings = tf.constant([[0, 0], [0, 16 - train_node1.shape[1]]])
        node1_tensor = tf.pad(train_node1, node1_paddings, mode='CONSTANT')
        node2_paddings = tf.constant([[0, 0], [0, 16 - train_node2.shape[1]]])
        node2_tensor = tf.pad(train_node2, node2_paddings, mode='CONSTANT')

        eval_node_padding = tf.constant([[0, 0], [0, 16 - eval_nodes.shape[1]]])
        eval_nodes = tf.pad(eval_nodes, eval_node_padding, mode='CONSTANT')
        eval_label_padding = tf.constant([[0, 0], [0, 16 - eval_labels.shape[1]]])
        eval_labels = tf.pad(eval_labels, eval_label_padding, mode='CONSTANT')
        eval_node1_padding = tf.constant([[0, 0], [0, 16 - eval_node1.shape[1]]])
        eval_node1 = tf.pad(eval_node1, eval_node1_padding, mode='CONSTANT')
        eval_node2_paddings = tf.constant([[0, 0], [0, 16 - eval_node2.shape[1]]])
        eval_node2 = tf.pad(eval_node2, eval_node2_paddings, mode='CONSTANT')
        print('\nTrain Tensor shapes (nodes, labels, node1, node2, target) : ')
        print(node_tensor.shape, label_tensor.shape, node1_tensor.shape, node2_tensor.shape, train_tgt_tensor.shape)
        print('\nEval Tensor shapes (nodes, labes, node1, node2) : ')
        print(eval_nodes.shape, eval_labels.shape, eval_node1.shape, eval_node2.shape)

        BUFFER_SIZE = len(train_tgt_tensor)
        BATCH_SIZE = args.batch_size
        steps_per_epoch = len(train_tgt_tensor) // BATCH_SIZE
        vocab_size = len(vocab.word_index) + 1
        dataset_size = train_tgt_tensor.shape[0]

        dataset = tf.data.Dataset.from_tensor_slices((node_tensor, label_tensor,
                                                        node1_tensor, node2_tensor, train_tgt_tensor)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        eval_set = tf.data.Dataset.from_tensor_slices((eval_nodes, eval_labels,
                                                    eval_node1, eval_node2))
        eval_set = eval_set.batch(BATCH_SIZE, drop_remainder=True)

        return (dataset, eval_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
                vocab_size, vocab, max_length_targ, dataset_size)
    
    else:
        (train_adj, train_node_tensor, train_role_tensor, 
        train_edges_tensor, train_tgt_tensor, eval_adj, eval_node_tensor, 
        eval_role_tensor, eval_edges_tensor, vocab, max_length) = load_gat_dataset(args.train_path, args.eval_path, 
                                                                                   args.vocab_path, args.opt, args.lang)
        node_padding = tf.constant([[0, 0], [0, 16-train_node_tensor.shape[1]]])
        node_tensor = tf.pad(train_node_tensor, node_padding, mode='CONSTANT')
        role_padding = tf.constant([[0, 0], [0, 16-train_role_tensor.shape[1]]])
        role_tensor = tf.pad(train_role_tensor, role_padding, mode='CONSTANT')
        edges_paddings = tf.constant([[0, 0], [0, 16 - train_edges_tensor.shape[1]]])
        edge_tensor = tf.pad(train_edges_tensor, edges_paddings, mode='CONSTANT')

        eval_node_padding = tf.constant([[0, 0], [0, 16-eval_node_tensor.shape[1]]])
        eval_node_tensor = tf.pad(eval_role_tensor, eval_node_padding, mode='CONSTANT')
        eval_role_padding = tf.constant([[0, 0], [0, 16-eval_role_tensor.shape[1]]])
        eval_role_tensor = tf.pad(eval_role_tensor, eval_role_padding, mode='CONSTANT')
        eval_edges_paddings = tf.constant([[0, 0], [0, 16 - eval_edges_tensor.shape[1]]])
        eval_edge_tensor = tf.pad(eval_edges_tensor, eval_edges_paddings, mode='CONSTANT')

        print('\nTrain Tensor shapes (nodes, roles, edges, target) : ')
        print(node_tensor.shape, role_tensor.shape, edge_tensor.shape, train_tgt_tensor.shape)
        print('\nEval Tensor shapes (nodes, roles, edges) : ')
        print(eval_node_tensor.shape, eval_role_tensor.shape, eval_edge_tensor.shape)
        
        BUFFER_SIZE = len(train_tgt_tensor)
        BATCH_SIZE = args.batch_size
        steps_per_epoch = len(train_tgt_tensor) // BATCH_SIZE
        vocab_size = len(vocab.word_index) + 1
        dataset_size = train_tgt_tensor.shape[0]

        #convert adj list to numpy array 
        train_adj = np.array(train_adj) 

        dataset = tf.data.Dataset.from_tensor_slices((train_adj, node_tensor,
                                                        role_tensor, edge_tensor, train_tgt_tensor)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        eval_set = tf.data.Dataset.from_tensor_slices((eval_adj, eval_node_tensor, 
                                                        eval_role_tensor, eval_edge_tensor))
        eval_set = eval_set.batch(BATCH_SIZE, drop_remainder=True)

        return (dataset, eval_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
                vocab_size, vocab, max_length, dataset_size)