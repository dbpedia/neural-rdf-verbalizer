"""
Inference, take a triple set, load the model and return the sentence
"""
import tensorflow as tf
from tqdm import tqdm
import pickle
import os
import sentencepiece as spm

from src.models import GraphAttentionModel, Transformer
from src.utils.model_utils import CustomSchedule
from src.DataLoader import GetGATDataset
from src.arguments import get_args

def LoadGatVocabs(args):
    with open('vocabs/gat/'+args.lang+'/'+args.opt+'_src_vocab', 'rb') as f:
        src_vocab = pickle.load(f)
    target_vocab = spm.SentencePieceProcessor()
    target_vocab.load('vocabs/' + args.model + '/' + args.lang + '/train_tgt.model')

    return src_vocab, target_vocab

def LoadModel(args):
    """
    Function to load the model from stored checkpoint.
    :param args: All arguments that were given to train file
    :type args: Argparse object
    :return: model
    :rtype: tf.keras.Model
    """
    # set up dirs
    if args.use_colab is None:
        output_file = 'results.txt'
        OUTPUT_DIR = 'ckpts'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    else:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/ckpts'
        output_file = OUTPUT_DIR + '/results.txt'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == "gat" and args.dec_type == "transformer":
        src_vocab, target_vocab = LoadGatVocabs(args)
        vocab_src_size = len(src_vocab.word_index) + 1
        vocab_tgt_size = target_vocab.get_piece_size()
        model = GraphAttentionModel.TransGAT(args, vocab_src_size, src_vocab,
                                            vocab_tgt_size, target_vocab)

    elif args.enc_type == 'transformer' and args.dec_type == 'transformer':
        num_layers = args.enc_layers
        num_heads = args.num_heads
        d_model = args.emb_dim
        dff = args.hidden_size
        dropout_rate = args.dropout
        vocab = LoadSeqVocabs(args.vocab_path)
        vocab_size = len(vocab.word_index) + 1
        model = Transformer.Transformer(args, vocab_size)

    if args.decay is not None:
        learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)
    else:
        optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)

    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    return model

if __name__ == "__main__":
    #Parse the arguments
    args = get_args()
    EvalResultsFile = 'predictions.txt'

    (dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE,
     steps_per_epoch, src_vocab_size, src_vocab, tgt_vocab_size,
     tgt_vocab, max_length_targ, dataset_size) = GetGATDataset(args)

    #load the vocabs
    src_vocab, tgt_vocab = LoadGatVocabs(args)
    model = LoadModel(args)
    results = []
    ref_target = []
    eval_results = open(EvalResultsFile, 'w+')
    for (batch, (nodes, labels, node1, node2)) in tqdm(enumerate(test_set)):
        predictions = model(nodes, labels, node1,
                            node2, targ=None, mask=None)
        pred = [(predictions['outputs'].numpy().tolist())]
        for i in range(len(pred[0])):
            sentence = (tgt_vocab.DecodeIds(list(pred[0][i])))
            sentence = sentence.partition("start")[2].partition("end")[0]
            eval_results.write(sentence + '\n')
            results.append(sentence)

    eval_results.close()