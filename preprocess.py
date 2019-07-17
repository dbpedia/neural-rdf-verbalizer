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
import pickle
import os

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
    '--path', type=str, required=False, help='Path to source.triple file')
parser.add_argument(
    '--train', type=bool, required=False, help='Preprocess train files or eval files')
parser.add_argument(
    '--opt', type=str, required=True, help='Adjacency processing or feature: adj -> adjacency matrix')
parser.add_argument(
    '--use_colab', type=bool, required=False, help='Use colab or not')
parser.add_argument(
    '--lang', type=str, required=True, help='Language of the dataset')

args = parser.parse_args()

def pre_process_with_roles(path):
    dest = open(path, 'r')
    count = 0
    for line in dest:
        g = nx.MultiDiGraph()
        temp_edge = []
        triple_list = line.split('< TSP >')
        for l in triple_list:
            l = l.strip().split(' | ')
            print(l)
            g.add_edge(l[0], l[1])
            g.add_edge(l[1], l[2])
            g.add_edge(l[0], l[2])
            temp_edge.append(l[1])
        print('-')
        edges.append(temp_edge)
        print(list(g.nodes()))
        nodes.append(list(g.nodes()))

        # set roles
        roles_ = []
        for node in list(g.nodes()):
            role = ''
            for l in triple_list:
                l = l.strip().split(' | ')

                if l[0] == node:
                    if role == 'object':
                        role = 'bridge'
                    else:
                        role = 'subject'
                elif l[1] == node:
                    role = 'predicate'
                elif l[2] == node:
                    if role == 'subject':
                        role = 'bridge'
                    else:
                        role = 'object'
            roles_.append(role)
        roles.append(roles_)

        print(roles_)

        array = nx.to_numpy_matrix(g)
        print(array)
        result = np.zeros((16, 16))

        result[:array.shape[0], :array.shape[1]] = array

        result += np.identity(16)

        adj.append(result)
        diag = np.sum(result, axis=0)
        D = np.matrix(np.diag(diag))
        degree_mat.append(D)
        result = D**-1 * result

    dest.close()

def pre_process(path, lang):
    dest = open(path, 'r')
    lang = '<'+lang+'>'
    for line in dest:
        g = nx.MultiDiGraph()
        temp_label = []
        temp_node1 = []
        temp_node2 = []
        triple_list = line.split('< TSP >')
        for l in triple_list:
            l = l.strip().split(' | ')
            l = [lang+' '+x for x in l]
            g.add_edge(l[0], l[1], label='A_ZERO')
            #g.add_edge(l[1], l[0])
            g.add_edge(l[1], l[2], label='A_ONE')
            #g.add_edge(l[2], l[1])
        node_list = list(g.nodes())
        print(node_list)
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
        adj = []
        degree_mat = []
        tensor = []
        nodes = []
        roles = []
        edges = []
        pre_process_with_roles(args.path)
        tensor = np.array(tensor)
        degree_mat = np.array(degree_mat)
        adj = np.array(adj)
        print(tensor.shape)
        if args.train is not None:
            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive')
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/train'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

                np.save('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_pure_adj', adj)
                np.save('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_degree_matrix', degree_mat)
                np.save('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_adj', tensor)
                with open('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_edges', 'wb') as fp:
                    pickle.dump(edges, fp)
                with open('/content/gdrive/My Drive/data/processed_graphs/train/train_node_roles', 'wb') as fp:
                    pickle.dump(roles, fp)
            else:
                OUTPUT_DIR = 'data/processed_graphs/train'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

                np.save('data/processed_graphs/train/train_graph_adj', tensor)
                np.save('data/processed_graphs/train/train_graph_pure_adj', adj)
                np.save('data/processed_graphs/train/train_graph_degree_matrix',degree_mat)
                with open('data/processed_graphs/train/train_graph_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open('data/processed_graphs/train/train_graph_edges', 'wb') as fp:
                    pickle.dump(edges, fp)
                with open('data/processed_graphs/train/train_node_roles', 'wb') as fp:
                    pickle.dump(roles, fp)
        else:
            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive')
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/eval'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

                np.save('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_pure_adj', adj)
                np.save('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_degree_matrix', degree_mat)
                np.save('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_adj', tensor)
                with open('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_edges', 'wb') as fp:
                    pickle.dump(edges, fp)
                with open('/content/gdrive/My Drive/data/processed_graphs/eval/eval_node_roles', 'wb') as fp:
                    pickle.dump(roles, fp)
            else:
                OUTPUT_DIR = 'data/processed_graphs/eval'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

                np.save('data/processed_graphs/eval/eval_graph_adj', tensor)
                np.save('data/processed_graphs/eval/eval_graph_pure_adj', adj)
                np.save('data/processed_graphs/eval/eval_graph_degree_matrix', degree_mat)
                with open('data/processed_graphs/eval/eval_graph_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open('data/processed_graphs/eval/eval_graph_edges', 'wb') as fp:
                    pickle.dump(edges, fp)
                with open('data/processed_graphs/eval/eval_node_roles', 'wb') as fp:
                    pickle.dump(roles, fp)

    elif args.opt == 'reif':
        nodes = []
        node1 = []
        node2 = []
        labels = []
        pre_process(args.path, args.lang)
        if args.train is not None:
            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive')
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/train/'+args.lang
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
                with open(OUTPUT_DIR+'/train_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open(OUTPUT_DIR+'/train_node1', 'wb') as fp:
                    pickle.dump(node1, fp)
                with open(OUTPUT_DIR+'/train_node2', 'wb') as fp:
                    pickle.dump(node2, fp)
                with open(OUTPUT_DIR+'/train_labels', 'wb') as fp:
                    pickle.dump(labels, fp)
            else:
                OUTPUT_DIR = 'data/processed_graphs/train/'+args.lang
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
                with open(OUTPUT_DIR+'/train_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open(OUTPUT_DIR+'/train_node1', 'wb') as fp:
                    pickle.dump(node1, fp)
                with open(OUTPUT_DIR+'/train_node2', 'wb') as fp:
                    pickle.dump(node2, fp)
                with open(OUTPUT_DIR+'/train_labels', 'wb') as fp:
                    pickle.dump(labels, fp)
        else:
            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive')
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/'+args.lang+'/eval'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
                with open(OUTPUT_DIR+'/eval_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open(OUTPUT_DIR+'/eval_node1', 'wb') as fp:
                    pickle.dump(node1, fp)
                with open(OUTPUT_DIR+'/eval_node2', 'wb') as fp:
                    pickle.dump(node2, fp)
                with open(OUTPUT_DIR+'/eval_labels', 'wb') as fp:
                    pickle.dump(labels, fp)
            else:
                OUTPUT_DIR = 'data/processed_graphs/'+args.lang+'/eval'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
                with open(OUTPUT_DIR+'/eval_nodes', 'wb') as fp:
                    pickle.dump(nodes, fp)
                with open(OUTPUT_DIR+'/eval_node1', 'wb') as fp:
                    pickle.dump(node1, fp)
                with open(OUTPUT_DIR+'/eval_node2', 'wb') as fp:
                    pickle.dump(node2, fp)
                with open(OUTPUT_DIR+'eval_labels', 'wb') as fp:
                    pickle.dump(labels, fp)