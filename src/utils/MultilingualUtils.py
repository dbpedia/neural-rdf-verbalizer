'''
Utils file which has all generic dataloader functions for mulilingual model
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import networkx as nx

import pickle
import sentencepiece as spm
from pathlib import Path

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