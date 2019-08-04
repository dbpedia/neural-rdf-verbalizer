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

from src.utils.model_utils import PreProcessSentence
from src.utils.PreprocessingUtils import PreProcess, PreProcessRolesModel
import sentencepiece as spm
import argparse
import pickle
import io
import os

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
    '--train_src', type=str, required=False, help='Path to train source file')
parser.add_argument(
    '--train_tgt', type=str, required=False, help='Path to train target file ')
parser.add_argument(
    '--eval_src', type=str, required=False, help='Path to eval source file')
parser.add_argument(
    '--eval_tgt', type=str, required=False, help='Path to eval target file')
parser.add_argument(
    '--test_src', type=str, required=False, help='Path to test source file')
parser.add_argument(
    '--model', type=str, required=True, help='Preprocess for GAT model or seq2seq model')
parser.add_argument(
    '--opt', type=str, required=True, help='Role processing or Reification: role, reif ')
parser.add_argument(
    '--use_colab', type=bool, required=False, help='Use colab or not')
parser.add_argument(
    '--lang', type=str, required=True, help='Language of the dataset')
parser.add_argument(
    '--vocab_size', type=int, required=False, help='Size of target vocabulary')

args = parser.parse_args()

#intialise sentence piece
def TrainVocabs(args):
    os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
    # exception check for 'None' value of vocab_size
    # Vocab size is not used during inference, only
    # during training and eval preprocessing
    try:
        if args.vocab_size is None:
            raise ValueError
        if args.use_colab is not None:
            spm.SentencePieceTrainer.Train('--input=' + args.train_tgt +','+ args.eval_tgt + ' \
                                            --model_prefix=/content/gdrive/My Drive/data/vocabs/'+args.model+'/'+args.lang+'/train_tgt \
                                            --vocab_size='+str(
                args.vocab_size)+' --character_coverage=1.0 --model_type=bpe')
        else:
            spm.SentencePieceTrainer.Train('--input=' + args.train_tgt + ',' + args.eval_tgt + ' \
                                                        --model_prefix=vocabs/' + args.model + '/' + args.lang + '/train_tgt \
                                                        --vocab_size=' + str(
                args.vocab_size) + ' --character_coverage=1.0 --model_type=bpe')
    except ValueError:
        print('Please enter the vocab size to'
              'train the SentencePiece vocab')
        exit(0)

    sp = spm.SentencePieceProcessor()
    sp.load('vocabs/'+args.model+'/'+args.lang+'/train_tgt.model')
    print('Sentencepiece vocab size {}'.format(sp.get_piece_size()))

    return sp

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
            test_nodes, test_labels, test_node1, test_node2 = PreProcess(args.test_src, args.lang)

            # Build and save the vocab
            print('Building the  Source Vocab file... ')
            train_tgt = io.open(args.train_tgt, encoding='UTF-8').read().strip().split('\n')
            train_tgt = [PreProcessSentence(w, args.lang) for w in train_tgt]
            #vocab_train_tgt = [tokenizer(w) for w in train_tgt]
            eval_tgt = io.open(args.eval_tgt, encoding='UTF-8').read().strip().split('\n')
            eval_tgt = [PreProcessSentence(w, args.lang) for w in eval_tgt]

            vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
            vocab.fit_on_texts(train_nodes)
            vocab.fit_on_texts(train_labels)
            vocab.fit_on_texts(train_node1)
            vocab.fit_on_texts(train_node2)
            vocab.fit_on_texts(eval_nodes)
            vocab.fit_on_texts(eval_labels)
            vocab.fit_on_texts(eval_node1)
            vocab.fit_on_texts(eval_node2)
            print('Vocab Size : {}\n'.format(len(vocab.word_index)))

            train_input = list(zip(train_nodes, train_labels, train_node1, train_node2))
            eval_input = list(zip(eval_nodes, eval_labels, eval_node1, eval_node2))
            test_input = list(zip(test_nodes, test_labels, test_node1, test_node2))
            train_set = list(zip(train_input, train_tgt))
            eval_set = list(zip(eval_input, eval_tgt))
            print('Train and eval dataset size : {} {} '.format(len(train_set), len(eval_set)))

            if args.use_colab is not None:
                from google.colab import drive

                drive.mount('/content/gdrive', force_remount=True)
                OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/'+args.lang+'/'+args.model
                if not os.path.isdir(OUTPUT_DIR):
                     os.makedirs(OUTPUT_DIR)
                # save the vocab file
                os.makedirs(('/content/gdrive/My Drive/data/vocabs/gat/' + args.lang), exist_ok=True)
                with open(('/content/gdrive/My Drive/data/vocabs/gat/' + args.lang + '/' + args.opt + '_src_vocab'), 'wb+') as fp:
                    pickle.dump(vocab, fp)

            else:
                OUTPUT_DIR = 'data/processed_graphs/'+args.lang+'/'+args.model
                if not os.path.isdir(OUTPUT_DIR): 
                    os.makedirs(OUTPUT_DIR)
                # save the vocab file
                os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
                with open(('vocabs/gat/' + args.lang + '/' + args.opt + '_src_vocab'), 'wb+') as fp:
                    pickle.dump(vocab, fp)

            print('Vocab file saved !\n')
            print('Preparing the Graph Network datasets...')

            with open(OUTPUT_DIR+'/'+args.opt+'_train', 'wb') as fp:
                pickle.dump(train_set, fp)
            with open(OUTPUT_DIR+'/'+args.opt+'_eval', 'wb') as fp:
                pickle.dump(eval_set, fp)
            with open(OUTPUT_DIR+'/'+args.opt+'_test', 'wb') as fp:
                pickle.dump(test_input, fp)
            print('Dumped the train, eval and test datasets.')
                
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
        test_src = io.open(args.test_src, encoding='UTF-8').read().strip().split('\n')
        test_src = [PreProcessSentence(w, args.lang) for w in test_src]

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
        with open(OUTPUT_DIR+'/'+'test', 'wb') as fp:
            pickle.dump(test_src, fp)
        print('Dumped the train and eval datasets.')