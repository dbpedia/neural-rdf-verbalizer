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
import unicodedata
import pickle
import re
import io
import os

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
    '--train_src', type=str, required=True, help='Path to train source file')
parser.add_argument(
    '--train_tgt', type=str, required=True, help='Path to train target file ')
parser.add_argument(
    '--eval_src', type=str, required=True, help='Path to eval source file')
parser.add_argument(
    '--eval_tgt', type=str, required=True, help='Path to eval target file')
parser.add_argument(
    '--model', type=str, required=True, help='Preprocess for GAT model or seq2seq model')
parser.add_argument(
    '--opt', type=str, required=True, help='Adjacency processing or feature: adj -> adjacency matrix')
parser.add_argument(
    '--use_colab', type=bool, required=False, help='Use colab or not')
parser.add_argument(
    '--lang', type=str, required=True, help='Language of the dataset')

args = parser.parse_args()

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w, lang):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    if lang== 'eng':
        w = re.sub(r"[^0-9a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w
    return w

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
        train_nodes.append(list(g.nodes()))

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
        triple_list = line.split('< TSP >')
        for l in triple_list:
            l = l.strip().split(' | ')
            #l = [lang+' '+x for x in l]
            g.add_edge(l[0], l[1], label='A_ZERO')
            #g.add_edge(l[1], l[0])
            g.add_edge(l[1], l[2], label='A_ONE')
            #g.add_edge(l[2], l[1])
        node_list = list(g.nodes())
        #node_list.append(lang)
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
    if args.model == 'gat':
        if args.opt == 'adj':
            adj = []
            degree_mat = []
            tensor = []
            train_nodes = []
            roles = []
            edges = []
            pre_process_with_roles(args.path)
            tensor = np.array(tensor)
            degree_mat = np.array(degree_mat)
            adj = np.array(adj)
            print(tensor.shape)
            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive')
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/train'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

                np.save('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_pure_adj', adj)
                np.save('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_degree_matrix', degree_mat)
                np.save('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_adj', tensor)
                with open('/content/gdrive/My Drive/data/processed_graphs/train/train_graph_nodes', 'wb') as fp:
                    pickle.dump(train_nodes, fp)
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
                    pickle.dump(train_nodes, fp)
                with open('data/processed_graphs/train/train_graph_edges', 'wb') as fp:
                    pickle.dump(edges, fp)
                with open('data/processed_graphs/train/train_node_roles', 'wb') as fp:
                    pickle.dump(roles, fp)

            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive')
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/eval'
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

                np.save('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_pure_adj', adj)
                np.save('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_degree_matrix', degree_mat)
                np.save('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_adj', tensor)
                with open('/content/gdrive/My Drive/data/processed_graphs/eval/eval_graph_nodes', 'wb') as fp:
                    pickle.dump(train_nodes, fp)
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
                    pickle.dump(train_nodes, fp)
                with open('data/processed_graphs/eval/eval_graph_edges', 'wb') as fp:
                    pickle.dump(edges, fp)
                with open('data/processed_graphs/eval/eval_node_roles', 'wb') as fp:
                    pickle.dump(roles, fp)

        elif args.opt == 'reif':
            
            train_nodes, train_labels, train_node1, train_node2 = pre_process(args.train_src, args.lang)
            eval_nodes, eval_labels, eval_node1, eval_node2 = pre_process(args.eval_src, args.lang)

            # Build and save the vocab
            print('Building the Vocab file... ')
            train_tgt = io.open(args.train_tgt, encoding='UTF-8').read().strip().split('\n')
            train_tgt = [preprocess_sentence(w, args.lang) for w in train_tgt]
            eval_tgt = io.open(args.eval_tgt, encoding='UTF-8').read().strip().split('\n')
            eval_tgt = [preprocess_sentence(w, args.lang) for w in eval_tgt]

            vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
            vocab.fit_on_texts(train_tgt)
            vocab.fit_on_texts(eval_tgt)
            vocab.fit_on_texts(train_nodes)
            vocab.fit_on_texts(train_labels)
            vocab.fit_on_texts(train_node1)
            vocab.fit_on_texts(train_node2)
            vocab.fit_on_texts(eval_nodes)
            vocab.fit_on_texts(eval_labels)
            vocab.fit_on_texts(eval_node1)
            vocab.fit_on_texts(eval_node2)
            print('Vocab Size : {}\n'.format(len(vocab.word_index)))

            #save the vocab file
            os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
            with open(('vocabs/gat/' + args.lang + '/vocab'), 'wb+') as fp:
                pickle.dump(vocab, fp)

            print('Vocab file saved !\n')
            print('Preparing the Graph Network datasets...')
            train_input = list(zip(train_nodes, train_labels, train_node1, train_node2))
            eval_input = list(zip(eval_nodes, eval_labels, eval_node1, eval_node2))
            train_set = list(zip(train_input, train_tgt))
            eval_set = list(zip(eval_input, eval_tgt))
            print('Train and eval dataset size : {} {} '.format(len(train_set), len(eval_set)))

            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive')
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/'+args.lang
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
                with open(OUTPUT_DIR+'/train', 'wb') as fp:
                    pickle.dump(train_set, fp)
                with open(OUTPUT_DIR+'/eval', 'wb') as fp:
                    pickle.dump(eval_set, fp)

            else:
                OUTPUT_DIR = 'data/processed_graphs/'+args.lang
                if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
                with open(OUTPUT_DIR+'/train', 'wb') as fp:
                    pickle.dump(train_set, fp)
                with open(OUTPUT_DIR+'/eval', 'wb') as fp:
                    pickle.dump(eval_set, fp)
            print('Dumped the train and eval datasets.')
                
    else:
        print('hello')