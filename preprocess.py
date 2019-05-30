""" Script to preprocess the .triple files to
create graphs using the networkx library, and
save the adjacency matrix as numpy array

 Takes in the .triple file which has each RDF triple
 from the dataset in <subject | predicate | object>
 form. Also uses networkx library to represent each
 example instance s a graph and get the adjacency matrix
 as a numpy array or a tf.Tensor

 Then creates a dataset file with each entry being the
 adjacency matrix of that graph, combined with nodes of
 the graph whose embeddings are used as node features
 are used as inputs to the Graph Neural Network encoder.
"""
import tensorflow as tf

import numpy as np
import networkx as nx
import argparse

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
    '--path', type=str, required=False, help='Path to source.triple file')
parser.add_argument(
    '--opt', type=str, required=True, help='Adjacency processing or feature: adj -> adjacency matrix')

args = parser.parse_args()


def pre_process(path):
    dest = open(path, 'r')
    count = 0
    for line in dest:
        g = nx.MultiDiGraph()

        triple_list = line.split('< TSP >')
        for l in triple_list:
            l = l.strip().split(' | ')
            print(l)
            g.add_edge(l[0], l[2], label=l[1])
        nodes.append(list(g.nodes))
        array = nx.to_numpy_array(g)
        print(array)
        result = np.zeros((16, 16))
        result[:array.shape[0], :array.shape[1]] = array
        tensor.append(result)

    dest.close()


def find_embedding(embedding, str):
    """Function to look up the BERT embeddings of words

    Arguments:
        path {str} -- path to the array containing all nodes in a graph
    Returns:
        tf.Tensor -- node feature matrix
    """
    result = embedding(str)
    emb = []
    for let in result:
        emb.append(let[1][0])
    result = tf.math.accumulate_n(emb)

    return result


def node_tensors(nodes, embedding):
    """Function to create feature matrix for each graph

    Arguments:
        nodes {.npy} -- Numpy array that has list of all nodes in each
        training example
        embedding {Bert_embedding object} -- BERT model object
    """
    result = []
    for node in nodes:
        tensor = []
        emb = embedding(node)
        for ele in emb:
            tensor.append(ele[1][0])
        tensor = np.array(tensor)
        result.append(tensor)
    result = np.array(result)

    return result


if __name__ == '__main__':
    if args.opt == 'adj':
        tensor = []
        nodes = []
        pre_process(args.path)
        tensor = np.array(tensor)
        print(tensor.shape)
        np.save('data/graph_adj', tensor)
        np.save('data/graph_nodes', nodes)
    else:
        if args.emb != 768:
            model = "bert_24_1024_16"
            dataset_name = 'book_corpus_wiki_en_cased'
            # initialize the embedding model
            embedding = BertEmbedding(model=model, dataset_name=dataset_name)
        else:
            embedding = BertEmbedding()
        nodes = np.load('data/graph_nodes.npy')
        result = node_tensors(nodes, embedding)
        np.save('data/graph_features.npy', result)





