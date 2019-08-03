'''
Utils file which has all generic dataloader functions for mulilingual model
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import networkx as nx

import pickle
import os
from src.models.GraphAttentionModel import TransGAT
import sentencepiece as spm
from pathlib import Path

languages = ['eng']

def LoadTeacherModels ():
    for lang in languages:
        # load Trained teacher model parameters
        with open(lang + '_model_params', 'rb') as fp:
            params= pickle.load(fp)
                    
        args = params['args'] 

        if args.use_colab is None:
            OUTPUT_DIR = 'ckpts/' + args.lang
            if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
        else:
            from google.colab import drive

            drive.mount('/content/gdrive')
            OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/' + args.lang
            if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

        if args.enc_type == 'gat' and args.dec_type == 'transformer':
            models = {}
            OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type
            
            # Load the vocabs
            with open('vocabs/'+args.model+'/'+
                    lang+'/'+args.opt+'_src_vocab', 'rb') as fp:
                src_vocab = pickle.load(fp)
            #loading the target vocab
            sp = spm.SentencePieceProcessor()
            sp.load('vocabs/'+args.model+'/'+
                    lang+'/'+'train_tgt.model')

            print('Loaded '+lang +' Parameters..')
            models[lang+'_model'] = TransGAT(params['args'], params['src_vocab_size'], src_vocab,
                                            params['tgt_vocab_size'], sp)
            # Load the latest checkpoints
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                            epsilon=1e-9)

            ckpt = tf.train.Checkpoint(
                model=models[lang+'_model'],
                optimizer=optimizer
            )

            ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
            if ckpt_manager.latest_checkpoint:
                ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

        print('Loaded All Teacher models !')
        return models

def Padding(tensor, max_length):
    padding = tf.constant([[0, 0], [0, max_length - tensor.shape[1]]])
    padded_tensor = tf.pad(tensor, padding, mode='CONSTANT')

    return padded_tensor

def PreProcess(path, lang):
    nodes = []
    labels = []
    node1 = []
    node2 = []
    dest = open(path, 'r')
    lang = '<'+lang+'>'
    for line in dest:
        g = nx.MultiDiGraph()
        temp_label = []
        temp_node1 = []
        temp_node2 = []
        triple_list = line.split('<TSP>')
        #triple_list = triple_list[:-1]
        for l in triple_list:
            l = l.strip().split(' | ')
            #l = [lang+' '+x for x in l]
            g.add_edge(l[0], l[1], label='A_ZERO')
            #g.add_edge(l[1], l[0])
            g.add_edge(l[1], l[2], label='A_ONE')
            #g.add_edge(l[2], l[1])
        node_list = list(g.nodes())
        node_list.append(lang)
        #print(node_list)
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