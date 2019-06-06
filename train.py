""" Script to train the selected model """

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
import GPUtil as GPU
GPUs = GPU.getGPUs()
gpu = GPUs[0]

from data_loader import get_dataset, get_gat_dataset, convert, preprocess_sentence
from src.models import model_params
from src.layers.attention_layer import BahdanauAttention
from src.layers.encoders import GraphEncoder
from src.layers.decoders import Decoder
from src.models import transformer
from arguments import get_args

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))


def loss_function(real, pred, loss_object):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask 

    return tf.reduce_mean(loss_)

if __name__ == "__main__":
    args = get_args()

    #set up dirs
    if args.use_colab == False:
        args.checkpoint_dir = 'ckpts/' if args.checkpoint_dir is None else args.checkpoint_dir
        output_file = 'results.txt'
        if args.output_dir is None and not os.path.isdir('ckpts'):
            os.mkdir('ckpts')
    else:
        from google.colab import drive
        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/ckpts'
        args.checkpoint_dir = OUTPUT_DIR
        output_file = OUTPUT_DIR + '/results.txt'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    if args.enc_type == 'gat':
        (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, 
        vocab_tgt_size, vocab_nodes_size, target_lang, max_length_targ) = get_gat_dataset(args)

        embedding = tf.keras.layers.Embedding(vocab_nodes_size, args.emb_dim) 
        
        encoder = GraphEncoder(args)
        decoder = Decoder(vocab_tgt_size, args.emb_dim, args.enc_units, BATCH_SIZE)

        optimizer = tf.train.AdamOptimizer()
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        checkpoint_dir = args.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint( optimizer=optimizer,
                                            encoder= encoder,
                                            decoder= decoder)
        if args.epochs is not None:
            steps = args.epochs * steps_per_epoch
        else:
            steps = args.steps

        def train_step(adj, nodes, targ):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(nodes, adj, encoder.trainable)
                dec_input=tf.expand_dims([target_lang.word_index['<start>']] * BATCH_SIZE, 1)

                # Apply teacher forcing 
                for t in range(1, targ.shape[1]):
                    # pass encoder output to decoder
                    predictions, dec_hidden, _ = decoder(dec_input, enc_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions, loss_object)

                    #using teacher forcing 
                    dec_input = tf.expand_dims(targ[:, t], 1) 

            batch_loss = (loss / int(targ.shape[1]))
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables) 

            optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss   

        # Eval function
        def eval_step(adj, nodes, targ):
            # set encoder and decoder to eval state
            encoder.trainable = False
            decoder.trainable = False

            eval_loss = 0
            enc_output, enc_hidden = encoder(nodes, adj, encoder.trainable)
            dec_input = tf.expand_dims([target_lang.word_index['<start>']] * BATCH_SIZE, 1)
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, enc_hidden, enc_output)
                eval_loss += loss_function(targ[:, t], predictions, loss_object)
                dec_input = tf.expand_dims(targ[:, t], 1)

            eval_loss = (eval_loss / int(targ.shape[1]))

            encoder.trainable = True
            decoder.trainable = True

            return eval_loss

        total_loss =0
        for (batch, (adj, nodes, edges, targ)) in enumerate(dataset.take(steps)):
            start = time.time()

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
                print('Step {} Eval Loss{:.4f}'.format(batch,
                                                        eval_loss.numpy()))
            else:
                batch_loss = train_step(adj, nodes, targ)
                print('Step {} Train Loss{:.4f}'.format(batch,
                                                        batch_loss.numpy()))
            #printm()

            if batch % args.checkpoint == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                print('Time {} \n'.format(time.time() - start))



    else:
        dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, vocab_inp_size, vocab_tgt_size = get_dataset(args)
        example_input_batch, example_target_batch= next(iter(dataset))






