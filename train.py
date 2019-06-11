""" Script to train the selected model
    Used to train a single language model ( Teacher model ) 
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import argparse
import os
import time
import io
import tempfile
from six.moves import xrange
from absl import app as absl_app
from absl import flags
import psutil
import humanize
import os

from data_loader import get_dataset, get_gat_dataset, convert, preprocess_sentence
from src.models import model_params, transformer, graph_attention_model, rnn_model
from src.utils.model_utils import loss_function, model_summary
from src.layers.attention_layer import BahdanauAttention
from src.layers.encoders import GraphEncoder, RNNEncoder
from src.layers.decoders import RNNDecoder
from src.models import transformer
from arguments import get_args

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

if __name__ == "__main__":
    args = get_args()

    #set up dirs
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

    if args.enc_type == 'gat':
        OUTPUT_DIR += '/'+args.enc_type
        (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, 
        vocab_tgt_size, vocab_nodes_size, target_lang, max_length_targ) = get_gat_dataset(args)

        embedding = tf.keras.layers.Embedding(vocab_nodes_size, args.emb_dim) 
        model = graph_attention_model.GATModel(args, vocab_tgt_size, target_lang)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        checkpoint_dir = args.checkpoint_dir
        checkpoint = tf.train.Checkpoint( optimizer=optimizer,
                                            model=model)

        if args.epochs is not None:
            steps = args.epochs * steps_per_epoch
        else:
            steps = args.steps

        def train_step(adj, nodes, targ):
            loss = 0
            with tf.GradientTape() as tape:
                predictions, dec_hidden, loss = model(adj, nodes, targ)
            batch_loss =(loss / int(targ.shape[1]))
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

        # Eval function
        def eval_step(adj, nodes, targ):
            model.trainable = False
            eval_loss = 0
            predictions, dec_hidden, loss = model(adj, nodes, targ)
            eval_loss = (loss / int(targ.shape[1]))
            model.trainable = True

            return eval_loss

        total_loss =0
        for (batch, (adj, nodes, edges, targ)) in enumerate(dataset.take(steps)):
            start = time.time()

            if args.decay is not None:
                optimizer._lr = optimizer._lr * args.decay_rate **(batch // args.decay_steps)
            # type cast all tensors for uniformity
            adj = tf.cast(adj, tf.float32)
            nodes = tf.cast(nodes, tf.float32) 
            edges = tf.cast(edges, tf.float32) 
            targ = tf.cast(targ, tf.float32)

            #embed nodes 
            nodes = embedding(nodes)
            edges = embedding(edges)
            nodes = tf.add(nodes, edges)

            if batch % args.eval_steps == 0:
                eval_loss = eval_step(adj, nodes, targ)
                print('Step {} Eval Loss{:.4f} \n'.format(batch,
                                                        eval_loss.numpy()))
            else:
                batch_loss = train_step(adj, nodes, targ)
                print('Step {} Train Loss{:.4f} \n'.format(batch,
                                                        batch_loss.numpy()))

            if batch % args.checkpoint == 0:
                print("Saving checkpoint \n")
                checkpoint_prefix = os.path.join(OUTPUT_DIR, "ckpt")
                checkpoint.save(file_prefix=checkpoint_prefix)
            print('Time {} \n'.format(time.time() - start))

    else:
        OUTPUT_DIR += '/'+args.enc_type
        dataset, BUFFER_SIZE, BATCH_SIZE,\
        steps_per_epoch, vocab_inp_size, vocab_tgt_size, target_lang = get_dataset(args)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        model = rnn_model.RNNModel(vocab_inp_size, vocab_tgt_size, target_lang, args)
        enc_hidden = model.encoder.initialize_hidden_state()

        checkpoint_dir = args.checkpoint_dir
        checkpoint_prefix = os.path.join(OUTPUT_DIR, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=model)

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)

        def train_step(inp, targ, enc_hidden):
            loss = 0

            with tf.GradientTape() as tape:
                predictions, dec_hidden, loss = model(inp, targ, enc_hidden)

            batch_loss = (loss / int(targ.shape[1]))
            variables = model.trainable_variables 
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        def eval_step(inp, trg, enc_hidden):
            eval_loss = 0 
            model.trainable = False

            predictions, dec_hidden, eval_loss = model(inp, trg, enc_hidden)
            eval_loss = (eval_loss / int(targ.shape[1]))
            model.trainable = True

            return eval_loss

        for (batch, (inp, targ)) in enumerate(dataset.take(args.steps)):
            start = time.time()
            if args.decay is not None:
                optimizer._lr = optimizer._lr * args.decay_rate **(batch // args.decay_steps)

            if batch % args.eval_steps == 0:
                eval_loss = eval_step(inp, targ, enc_hidden)
                print('Step {} Eval Loss {:.4f} \n'.format(batch,eval_loss.numpy()))
            else:
                batch_loss = train_step(inp, targ, enc_hidden)
                print('Step {} Batch Loss {:.4f} \n'.format(batch,batch_loss.numpy()))

            if batch % args.checkpoint == 0:
                print("Saving checkpoint \n")
                checkpoint_prefix = os.path.join(OUTPUT_DIR, "ckpt")
                checkpoint.save(file_prefix=checkpoint_prefix)
            print('Time {} \n'.format(time.time() - start))

