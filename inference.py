"""
Inference, take a triple set, load the model and return the sentence
"""
import tensorflow as tf
import pickle
import os
from tqdm import tqdm
import sentencepiece as spm

from src.models import GraphAttentionModel
from src.utils.model_utils import Padding as padding
from src.DataLoader import GetGATDataset
from src.MultilingualDataLoader import ProcessMultilingualDataset
from src.arguments import get_args

def LoadGatVocabs(args):
    with open('vocabs/gat/'+args.lang+'/'+args.opt+'_src_vocab', 'rb') as f:
        src_vocab = pickle.load(f)
    if args.sentencepiece == 'True':
        target_vocab = spm.SentencePieceProcessor()
        target_vocab.load('vocabs/' + args.model + '/' + args.lang + '/train_tgt.model')
    else:
        target_vocab = src_vocab

    return src_vocab, target_vocab

def LoadModel(args):
    """
    Function to load the model from stored checkpoint.
    :param args: All arguments that were given to train file
    :type args: Argparse object
    :return: model
    :rtype: tf.keras.Model
    """

    if args.model=='gat':

        if args.use_colab is not None:
            OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/' + args.lang
            log_dir = OUTPUT_DIR + '/logs'
        else:
            log_dir = 'data/logs'
        with open(log_dir + '/' + args.lang + '_model_params', 'rb') as fp:
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
                      args.lang + '/' + model_args.opt + '_src_vocab', 'rb') as fp:
                src_vocab = pickle.load(fp)
            # loading the target vocab
            model_args.sentencepiece = 'False'
            if model_args.sentencepiece == 'True':
                sp = spm.SentencePieceProcessor()
                sp.load('vocabs/' + model_args.model + '/' +
                        args.lang + '/' + 'train_tgt.model')
                tgt_vocab = sp
            else:
                tgt_vocab = src_vocab

            print('Loaded ' + args.lang + ' Parameters..')
            model = GraphAttentionModel.TransGAT(params['args'], params['src_vocab_size'], src_vocab,
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

        print('Loaded ' + args.lang + ' model !')

        return model

if __name__ == "__main__":
    #Parse the arguments
    args = get_args()
    EvalResultsFile = 'predictions.txt'

    languages = ['eng', 'ger', 'rus']
    # load the vocabs
    src_vocab, tgt_vocab = LoadGatVocabs(args)
    model = LoadModel(args)

    if args.lang in languages:
        (test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
        src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab) = GetGATDataset(args, set='test')

        results = []
        ref_target = []
        reference = open(args.eval_ref, 'r')
        eval_results = open(EvalResultsFile, 'w+')

        for (batch, (nodes, labels, node1, node2)) in tqdm(enumerate(test_set)):
            predictions = model(nodes, labels, node1,
                                node2, targ=None, mask=None)
            pred = [(predictions['outputs'].numpy().tolist())]

            if args.sentencepiece == 'True':
                for i in range(len(pred[0])):
                    sentence = (tgt_vocab.DecodeIds(list(pred[0][i])))
                    sentence = sentence.partition("start")[2].partition("end")[0]
                    eval_results.write(sentence + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence)
            else:
                for i in pred:
                    sentences = tgt_vocab.sequences_to_texts(i)
                    sentence = [j.partition("start")[2].partition("end")[0] for j in sentences]
                    for w in sentence:
                        eval_results.write((w + '\n'))
                        ref_target.append(reference.readline())
                        results.append(w)

        eval_results.close()

    else:

        (dataset, src_vocab, src_vocab_size, tgt_vocab,
         tgt_vocab_size, MULTI_BUFFER_SIZE, steps_per_epoch, MaxSeqSize) = ProcessMultilingualDataset(args)

        results = []
        ref_target = []
        reference = open(args.eval_ref, 'r')
        eval_results = open(EvalResultsFile, 'w+')

        for (batch, (nodes, labels,
                     node1, node2, target)) in tqdm(enumerate(
            dataset['test_set'])):
            predictions = model(nodes, labels, node1,
                                node2, targ=None, mask=None)
            pred = [(predictions['outputs'].numpy().tolist())]

            if args.sentencepiece == 'True':
                for i in range(len(pred[0])):
                    sentence = (tgt_vocab.DecodeIds(list(pred[0][i])))
                    sentence = sentence.partition("start")[2].partition("end")[0]
                    eval_results.write(sentence + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence)
            else:
                for i in pred:
                    sentences = tgt_vocab.sequences_to_texts(i)
                    sentence = [j.partition("start")[2].partition("end")[0] for j in sentences]
                    for w in sentence:
                        eval_results.write((w + '\n'))
                        ref_target.append(reference.readline())
                        results.append(w)

        eval_results.close()