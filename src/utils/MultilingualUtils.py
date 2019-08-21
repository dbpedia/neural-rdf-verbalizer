'''
Utils file which has all generic dataloader functions for mulilingual model
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle

import sentencepiece as spm
import tensorflow as tf

from src.models.GraphAttentionModel import TransGAT


def LoadTeacherModels(lang):
    """
    Function to load the pre-trained teacher models
    and their parameters.

    :param lang: The language of the teacher model
    :type lang: str
    :return: The model object restored to latest checkpoint
    :rtype: tf.keras.models.Model object
    """

    # load Trained teacher model parameters
    log_dir = 'data/logs'
    with open(log_dir + '/' + lang + '_model_params', 'rb') as fp:
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
                  lang + '/' + model_args.opt + '_src_vocab', 'rb') as fp:
            src_vocab = pickle.load(fp)
        # loading the target vocab
        model_args.sentencepiece = 'False'
        if model_args.sentencepiece == 'True':
            sp = spm.SentencePieceProcessor()
            sp.load('vocabs/' + model_args.model + '/' +
                    lang + '/' + 'train_tgt.model')
            tgt_vocab = sp
        else:
            tgt_vocab = src_vocab

        print('Loaded ' + lang + ' Parameters..')
        model = TransGAT(params['args'], params['src_vocab_size'], src_vocab,
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

    print('Loaded ' + lang + ' Teacher model !')

    return model
