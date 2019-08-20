'''
Utils file which has all generic dataloader functions for mulilingual model
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle

import networkx as nx
import sentencepiece as spm
import tensorflow as tf

from src.models.GraphAttentionModel import TransGAT


def LoadTeacherModels(lang):
    """
    Function to load the pre-trained teacher models
    and their parameters.

    :param lang: The language of the teacher model
    :type lang: str
    :return: The model object restored to latest checkpoint
    :rtype: tf.keras.models.Model object
    """

    # load Trained teacher model parameters
    log_dir = 'data/logs'
    with open(log_dir + '/' + lang + '_model_params', 'rb') as fp:
        params = pickle.load(fp)

    model_args = params['args']

    if model_args.use_colab is None:
        OUTPUT_DIR = 'ckpts/' + model_args.lang
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    else:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/' + model_args.lang
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    if model_args.enc_type == 'gat' and model_args.dec_type == 'transformer':
        models = {}
        OUTPUT_DIR += '/' + model_args.enc_type + '_' + model_args.dec_type

        # Load the vocabs
        with open('vocabs/' + model_args.model + '/' +
                  lang + '/' + model_args.opt + '_src_vocab', 'rb') as fp:
            src_vocab = pickle.load(fp)
        # loading the target vocab
        model_args.sentencepiece = 'False'
        if model_args.sentencepiece == 'True':
            sp = spm.SentencePieceProcessor()
            sp.load('vocabs/' + model_args.model + '/' +
                    lang + '/' + 'train_tgt.model')
            tgt_vocab = sp
        else:
            tgt_vocab = src_vocab

        print('Loaded ' + lang + ' Parameters..')
        model = TransGAT(params['args'], params['src_vocab_size'], src_vocab,
                         params['tgt_vocab_size'], tgt_vocab)
        # Load the latest checkpoints
        optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)

        ckpt = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer
        )

        ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    print('Loaded ' + lang + ' Teacher model !')

    return model


def PreProcess(path, lang):
    """
    The preprocessing function that takes in a set of RDF triples
    and converts that into the graph dataset we will be using for
    our models.
    It loads the input from the disk, and iterates through each line
    of RDF triples

    ex - Let the triple set be
    " Dwarak | Loves | Physics <TSP> Dwarak | lives_in | India "
    There could be sets from one to seven triples.
    We intially extract each triple and create a triple list.
    triple 1 = Dwarak | loves | physics
    triple 2 = Dwaral | lives_in | India

    Then we create a Networkx Mutli-Di Graph object that creates
    a graph with all entities in the triples as nodes and describes the
    edge between them as connection of nodes.
    node 1 - Dwarak, node2 - loves, node3 - Physics
    Label of edge between node1 - node2 - A_ZERO
    Label of edge between node3 - node3 - A_ONE

    This way we impart the structural information of the triple set
    into the models inputs.

    :param path: The path to the RDF triple source file
    :type path: str
    :param lang: The language on which we are operating
    :type lang: str
    :return: nodes_list, node1 of edges, node2 of edges, and edge labels
    :rtype:list

    """
    nodes = []
    labels = []
    node1 = []
    node2 = []
    dest = open(path, 'r')
    lang = '<' + lang + '>'
    for line in dest:
        g = nx.MultiDiGraph()
        temp_label = []
        temp_node1 = []
        temp_node2 = []
        triple_list = line.split('<TSP>')
        # triple_list = triple_list[:-1]
        for l in triple_list:
            l = l.strip().split(' | ')
            # l = [lang+' '+x for x in l]
            g.add_edge(l[0], l[1], label='A_ZERO')
            # g.add_edge(l[1], l[0])
            g.add_edge(l[1], l[2], label='A_ONE')
            # g.add_edge(l[2], l[1])
        node_list = list(g.nodes())
        # node_list.append(lang)
        # print(node_list)
        nodes.append(node_list)
        edge_list = list(g.edges.data())
        for edge in edge_list:
            temp_node1.append(edge[0])
            temp_node2.append(edge[1])
            label = (edge[2]['label'])
            temp_label.append(label)
        node1.append(temp_node1)
        node2.append(temp_node2)
        labels.append(temp_label)

    dest.close()

    return nodes, labels, node1, node2
