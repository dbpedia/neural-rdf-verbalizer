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

import sentencepiece as spm
import numpy as np
import networkx as nx
import argparse
import unicodedata
import pickle
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
    '--opt', type=str, required=True, help='Role processing or Reification: role, reif ')
parser.add_argument(
    '--use_colab', type=bool, required=False, help='Use colab or not')
parser.add_argument(
    '--lang', type=str, required=True, help='Language of the dataset')

args = parser.parse_args()

#intialise sentence piece
def TrainVocabs(args):
    os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
    spm.SentencePieceTrainer.Train('--input=' + args.train_tgt +','+ args.eval_tgt + ' \
                                    --model_prefix=vocabs/'+args.model+'/'+args.lang+'/train_tgt \
                                    --vocab_size=10000 --character_coverage=1.0 --model_type=bpe')
    sp = spm.SentencePieceProcessor()
    sp.load('vocabs/'+args.model+'/'+args.lang+'/train_tgt.model')
    print('Sentencepiece vocab size {}'.format(sp.get_piece_size()))

    return sp

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def PreProcessSentence(w, lang):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
   # w = re.sub(r"([?.!,¿])", r" \1 ", w)
    #w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    #if lang== 'eng':
    #   w = re.sub(r"[^0-9a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = 'start ' + w + ' end'
    return w

def PreProcessRolesModel(path):
    adj = []
    degree_mat = []
    tensor = []
    train_nodes = []
    roles = []
    edges = []
    dest = open(path, 'r')
    count = 0
    for line in dest:
        g = nx.MultiDiGraph()
        temp_edge = []
        triple_list = line.split('< TSP >')
        for l in triple_list:
            l = l.strip().split(' | ')
            g.add_edge(l[0], l[1])
            g.add_edge(l[1], l[2])
            g.add_edge(l[0], l[2])
            temp_edge.append(l[1])
        edges.append(temp_edge)
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

        array = nx.to_numpy_matrix(g)
        result = np.zeros((16, 16))

        result[:array.shape[0], :array.shape[1]] = array

        result += np.identity(16)

        adj.append(result)
        diag = np.sum(result, axis=0)
        D = np.matrix(np.diag(diag))
        degree_mat.append(D)
        result = D**-1 * result

    dest.close()

    return adj, train_nodes, roles, edges

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
        if args.opt == 'role':
            print('Building the dataset...')
            train_adj, train_nodes, train_roles, train_edges = PreProcessRolesModel(args.train_src)
            eval_adj, eval_nodes, eval_roles, eval_edges = PreProcessRolesModel(args.eval_src)
            
            print('Building the Vocab file... ')
            train_tgt = io.open(args.train_tgt, encoding='UTF-8').read().strip().split('\n')
            train_tgt = [(PreProcessSentence(w, args.lang)) for w in train_tgt]
            eval_tgt = io.open(args.eval_tgt, encoding='UTF-8').read().strip().split('\n')
            eval_tgt = [(PreProcessSentence(w, args.lang)) for w in eval_tgt]

            #Create the train and test sets 
            train_input = list(zip(train_adj, train_nodes, train_roles, train_edges))
            eval_input = list(zip(eval_adj, eval_nodes, eval_roles, eval_edges)) 
            train_set = list(zip(train_input, train_tgt))
            eval_set = list(zip(eval_input, eval_tgt))
            print('Lengths of train and eval sets {} {}'.format(len(train_set), len(eval_set)))

            vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
            vocab.fit_on_texts(train_tgt)
            #vocab.fit_on_texts(eval_tgt)
            vocab.fit_on_texts(train_nodes)
            vocab.fit_on_texts(train_edges)
            vocab.fit_on_texts(train_roles)
            #vocab.fit_on_texts(eval_nodes)
            #vocab.fit_on_texts(eval_edges)
            #vocab.fit_on_texts(eval_roles)

            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive', force_remount=True)
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/'+args.lang+'/'+args.model
                if not os.path.isdir(OUTPUT_DIR): 
                    os.makedirs(OUTPUT_DIR)
            else:
                OUTPUT_DIR = 'data/processed_graphs/'+args.lang+'/'+args.model
                if not os.path.isdir(OUTPUT_DIR):
                    os.makedirs(OUTPUT_DIR)

            with open(OUTPUT_DIR+'/'+args.opt+'_train', 'wb') as fp:
                pickle.dump(train_set, fp)
            with open(OUTPUT_DIR+'/'+args.opt+'_eval', 'wb') as fp:
                pickle.dump(eval_set, fp)
            print('Vocab Size : {}\n'.format(len(vocab.word_index)))

            os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
            with open(('vocabs/gat/' + args.lang + '/'+args.opt+'_vocab'), 'wb+') as fp:
                pickle.dump(vocab, fp)

        elif args.opt == 'reif':

            vocab = TrainVocabs(args)

            train_nodes, train_labels, train_node1, train_node2 = PreProcess(args.train_src, args.lang)
            eval_nodes, eval_labels, eval_node1, eval_node2 = PreProcess(args.eval_src, args.lang)

            # Build and save the vocab
            print('Building the  Source Vocab file... ')
            train_tgt = io.open(args.train_tgt, encoding='UTF-8').read().strip().split('\n')
            train_tgt = [PreProcessSentence(w, args.lang) for w in train_tgt]
            #vocab_train_tgt = [tokenizer(w) for w in train_tgt]
            eval_tgt = io.open(args.eval_tgt, encoding='UTF-8').read().strip().split('\n')
            eval_tgt = [PreProcessSentence(w, args.lang) for w in eval_tgt]

            vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
            #vocab.fit_on_texts(train_tgt)
            #vocab.fit_on_texts(eval_tgt)
            vocab.fit_on_texts(train_nodes)
            vocab.fit_on_texts(train_labels)
            vocab.fit_on_texts(train_node1)
            vocab.fit_on_texts(train_node2)
            #vocab.fit_on_texts(eval_nodes)
            #vocab.fit_on_texts(eval_labels)
            #vocab.fit_on_texts(eval_node1)
            #vocab.fit_on_texts(eval_node2)
            print('Vocab Size : {}\n'.format(len(vocab.word_index)))

            #save the vocab file
            os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
            with open(('vocabs/gat/' + args.lang + '/'+args.opt+'_src_vocab'), 'wb+') as fp:
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

                drive.mount('/content/gdrive', force_remount=True)
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/'+args.lang+'/'+args.model
                if not os.path.isdir(OUTPUT_DIR):
                     os.makedirs(OUTPUT_DIR)

            else:
                OUTPUT_DIR = 'data/processed_graphs/'+args.lang+'/'+args.model
                if not os.path.isdir(OUTPUT_DIR): 
                    os.makedirs(OUTPUT_DIR)
            with open(OUTPUT_DIR+'/'+args.opt+'_train', 'wb') as fp:
                pickle.dump(train_set, fp)
            with open(OUTPUT_DIR+'/'+args.opt+'_eval', 'wb') as fp:
                pickle.dump(eval_set, fp)
            print('Dumped the train and eval datasets.')
                
    else:
        print('Building the dataset...')

        train_src = io.open(args.train_src, encoding='UTF-8').read().strip().split('\n')
        train_src = [PreProcessSentence(w, args.lang) for w in train_src]
        eval_src = io.open(args.eval_src, encoding='UTF-8').read().strip().split('\n')
        eval_src = [PreProcessSentence(w, args.lang) for w in eval_src]
        train_tgt = io.open(args.train_tgt, encoding='UTF-8').read().strip().split('\n')
        train_tgt = [PreProcessSentence(w, args.lang) for w in train_tgt]
        eval_tgt = io.open(args.eval_tgt, encoding='UTF-8').read().strip().split('\n')
        eval_tgt = [PreProcessSentence(w, args.lang) for w in eval_tgt]

        vocab = tf.keras.preprocessing.text.Tokenizer(filters='')        
        vocab.fit_on_texts(train_src)
        #vocab.fit_on_texts(eval_src)
        vocab.fit_on_texts(train_tgt)
        #vocab.fit_on_texts(eval_tgt)
        print('Vocab Size : {}\n'.format(len(vocab.word_index)))
        train_set = zip(train_src, train_tgt) 
        eval_set = zip(eval_src, eval_tgt) 
        
        #save the vocab file
        os.makedirs(('vocabs/seq2seq/' + args.lang), exist_ok=True)
        with open(('vocabs/seq2seq/' + args.lang + '/'+'vocab'), 'wb+') as fp:
            pickle.dump(vocab, fp)
        print('Vocab file saved !\n')
        if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive', force_remount=True)
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/'+args.lang+'/'+args.model
                if not os.path.isdir(OUTPUT_DIR):
                     os.makedirs(OUTPUT_DIR)

        else:
            OUTPUT_DIR = 'data/processed_graphs/'+args.lang+'/'+args.model
            if not os.path.isdir(OUTPUT_DIR): 
                os.makedirs(OUTPUT_DIR)
        with open(OUTPUT_DIR+'/'+'train', 'wb') as fp:
            pickle.dump(train_set, fp)
        with open(OUTPUT_DIR+'/'+'eval', 'wb') as fp:
            pickle.dump(eval_set, fp)
        print('Dumped the train and eval datasets.')