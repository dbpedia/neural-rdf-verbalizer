"""     Inference, take triple set as input and give sentence as output
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import networkx as nx
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from arguments import get_args
from src.models import graph_attention_model, rnn_model
from data_loader import get_gat_dataset, get_dataset, max_length, load_gat_dataset
from src.utils.model_utils import loss_function

def load_model(args):
    """
    Function to load the model from checkpoint
    :param args:
    :type args:
    :return: model and necessities 
    :rtype:
    """
    dir = args.checkpoint_dir
    og_dataset, BUFFER_SIZE, BATCH_SIZE,\
    steps_per_epoch, vocab_inp_size, vocab_tgt_size, target_lang = get_dataset(args)

    (graph_adj, node_tensor, nodes_lang, edge_tensor, edges_lang,
     target_tensor, target_lang, max_length_targ) = load_gat_dataset(args.graph_adj, args.graph_nodes,
                                                                     args.graph_edges, args.tgt_path, args.num_examples)
    model = graph_attention_model.GATModel(args, vocab_tgt_size, target_lang)
    optimizer = tf.train.AdamOptimizer()
    checkpoint = tf.train.Checkpoint( optimizer=optimizer,
                                            model=model)
    checkpoint.restore(args.checkpoint_dir)

    return model

def preprocess(line):
    """
    Function that preprocess the triples for evaluation
    :param sentence: triple set
    :type sentence: str
    :return: adjacency matrix, node list, edge list
    :rtype: dataset
    """
    edges = []
    nodes = []
    g = nx.MultiDiGraph()
    temp_edge = []
    triple_list = line.split('< TSP >')
    for l in triple_list:
        l = l.strip().split(' | ')
        print(l)
        g.add_edge(l[0], l[2], label=l[1])
        temp_edge.append(l[1])
    edges.append(temp_edge)
    nodes.append(list(g.nodes))
    array = nx.to_numpy_array(g)
    result = np.zeros((16, 16))
    result[:array.shape[0], :array.shape[1]] = array

    return result, nodes, edges

def tensor_process(args, nodes, edges):
    (graph_adj, node_tensor, nodes_lang, edge_tensor, edges_lang,
     target_tensor, target_lang, max_length_targ) = load_gat_dataset(args.graph_adj, args.graph_nodes,
                                                                     args.graph_edges, args.tgt_path, args.num_examples)
    node_tensor = nodes_lang.texts_to_sequences(nodes)
    node_tensor = tf.keras.preprocessing.sequence.pad_sequences(node_tensor, padding='post')
    edge_tensor = edges_lang.texts_to_sequences(edges)
    edge_tensor = tf.keras.preprocessing.sequence.pad_sequences(edge_tensor, padding='post')

    # Pad the node tensors tp 16 size
    print(node_tensor.shape, edge_tensor.shape)
    paddings = tf.constant([[0, 0], [0, 14]])
    node_tensor = tf.pad(node_tensor, paddings)
    # Pad the edge tensor to 16 size
    edge_paddings = tf.constant([[0, 0], [0, 15]])
    edge_tensor = tf.pad(edge_tensor, edge_paddings)
    vocab_nodes_size = len(nodes_lang.word_index) + 1

    embedding = tf.keras.layers.Embedding(vocab_nodes_size, args.emb_dim)
    node_tensor = embedding(node_tensor)
    edge_tensor = embedding(edge_tensor)

    return node_tensor, edge_tensor, target_lang, max_length_targ

if __name__ == "__main__":
    args = get_args()
    adj, nodes, edges = preprocess('Aarhus Airport | cityServed | Aarhus , Denmark')
    node, edge, targ_lang, max_length_targ = tensor_process(args, nodes, edges)
    model = load_model(args)
    model.trainable = False

    hidden = [tf.zeros((1, args.enc_units))]
    result = ''
    inputs = node + edge
    print(inputs.shape)
    adj = tf.cast(tf.expand_dims(adj, axis=0), dtype=tf.float32)
    inputs = tf.cast(inputs,  dtype=tf.float32)
    enc_output, enc_hidden = model.encoder(inputs, adj, False)
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    dec_hidden = enc_hidden

    for t in range(max_length_targ):
        predictions, dec_hidden, _ = model.decoder(dec_input, dec_hidden, enc_output)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + ' '
        if predicted_id != 0:
            if targ_lang.index_word[predicted_id] == '<end>':
                print(result)
        dec_input = tf.expand_dims([predicted_id], 0)

    print(result)












