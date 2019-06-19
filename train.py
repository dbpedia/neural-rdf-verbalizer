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
import os
from tqdm import tqdm 

from data_loader import get_dataset, get_gat_dataset
from src.models import model_params, transformer, graph_attention_model, rnn_model
from src.utils.model_utils import create_masks
from src.utils.model_utils import loss_function, model_summary
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

        if args.decay is not None:
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate,beta1=0.9, beta2=0.98, 
                                                epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98, 
                                                epsilon=1e-9)
        loss_object = tf.keras.losses.sparse_categorical_crossentropy

        ckpt = tf.train.Checkpoint(
            model = model,
            optimizer = optimizer
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            
        if args.learning_rate is not None:
            optimizer._lr = args.learning_rate

        if args.epochs is not None:
            steps = args.epochs * steps_per_epoch
        else:
            steps = args.steps

        def train_step(adj, nodes, edges, targ):
            print(optimizer._lr)
            loss = 0
            with tf.GradientTape() as tape:
                predictions, dec_hidden, loss = model(adj, nodes, edges, targ)
            batch_loss =(loss / int(targ.shape[1]))
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

        # Eval function
        def eval_step(adj, nodes, edges, targ):
            model.trainable = False
            eval_loss = 0
            predictions, dec_hidden, loss = model(adj, nodes, edges, targ)
            eval_loss = (loss / int(targ.shape[1]))
            model.trainable = True

            return eval_loss

        total_loss =0
        for epoch in range(args.epochs):
            with tqdm(total=(34352 // args.batch_size)) as pbar:
                for (batch, (adj, nodes, edges, targ)) in tqdm(enumerate(dataset)):
                    start = time.time()
                    # type cast all tensors for uniformity
                    adj = tf.cast(adj, tf.float32)
                    nodes = tf.cast(nodes, tf.float32) 
                    edges = tf.cast(edges, tf.float32) 
                    targ = tf.cast(targ, tf.float32)

                    #embed nodes 
                    nodes = embedding(nodes)
                    edges = embedding(edges)

                    if batch % args.eval_steps == 0:
                        eval_loss = eval_step(adj, nodes, edges, targ)
                        print('Batch {} Eval Loss{:.4f} '.format(batch,
                                                                eval_loss.numpy()))
                    else:
                        batch_loss = train_step(adj, nodes, edges, targ)
                        print('Batch {} Train Loss{:.4f} '.format(batch,
                                                                batch_loss.numpy()))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} \n'.format(time.time() - start))
                    pbar.update(1)
            if args.decay is not None:
                optimizer._lr = optimizer._lr * args.decay_rate ** (batch // 1)

    elif args.enc_type == 'rnn':
        OUTPUT_DIR += '/'+args.enc_type
        dataset, BUFFER_SIZE, BATCH_SIZE,\
        steps_per_epoch, vocab_inp_size, vocab_tgt_size, target_lang = get_dataset(args)

        if args.decay is not None:
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate,beta1=0.9, beta2=0.98, 
                                                epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98, 
                                                epsilon=1e-9)

        loss_object = tf.keras.losses.sparse_categorical_crossentropy
        model = rnn_model.RNNModel(vocab_inp_size, vocab_tgt_size, target_lang, args)
        enc_hidden = model.encoder.initialize_hidden_state()

        ckpt = tf.train.Checkpoint(
            model = model,
            optimizer = optimizer
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        if args.learning_rate is not None:
            optimizer._lr = args.learning_rate

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
        
        for epoch in range(args.epochs):
            with tqdm(total=(34352 // args.batch_size)) as pbar:
                for (batch, (inp, targ)) in tqdm(enumerate(dataset)):
                    start = time.time()

                    if batch % args.eval_steps == 0:
                        eval_loss = eval_step(inp, targ, enc_hidden)
                        print('Step {} Eval Loss {:.4f} '.format(batch,eval_loss.numpy()))
                    else:
                        batch_loss = train_step(inp, targ, enc_hidden)
                        print('Step {} Batch Loss {:.4f} '.format(batch,batch_loss.numpy()))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} '.format(time.time() - start))
                    pbar.update(1)
            if args.decay is not None:
                optimizer._lr = optimizer._lr * args.decay_rate ** (batch // 1)

    elif args.enc_type == 'transformer':
        OUTPUT_DIR += '/'+args.enc_type
        dataset, BUFFER_SIZE, BATCH_SIZE,\
        steps_per_epoch, vocab_inp_size, vocab_tgt_size, target_lang = get_dataset(args)
        num_layers = args.num_layers
        num_heads = args.num_heads
        d_model = args.emb_dim
        dff = args.hidden_size
        dropout_rate = args.dropout
        if args.epochs is None :
            epochs = args.steps // steps_per_epoch
        else:
            epochs = args.epochs
        
        if args.learning_rate is not None:
            learning_rate = args.learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.98, 
                                                epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98, 
                                                epsilon=1e-9)

        loss_object = tf.keras.losses.sparse_categorical_crossentropy
        model = transformer.Transformer(num_layers, d_model, num_heads, dff,
                          vocab_inp_size, vocab_tgt_size, dropout_rate)

        ckpt = tf.train.Checkpoint(
            model = model,
            optimizer = optimizer
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            
        if args.learning_rate is not None:
            optimizer._lr = args.learning_rate

        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = model(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
                loss = loss_function(tar_real, predictions, loss_object)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss

        def eval_step(inp, tar):
            model.trainable = False
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)


            predictions, _ = model(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
            loss = loss_function(tar_real, predictions, loss_object)
            model.trainable = True
            
            return loss

        for epoch in range(epochs):
            start = time.time()
            print("Learning rate "+str(optimizer._lr))
            with tqdm(total=(34352 // args.batch_size)) as pbar:
                for (batch, (inp, tar)) in tqdm(enumerate(dataset)):
                    if (batch % args.eval_steps == 0):
                        batch_loss = train_step(inp, tar)
                        print('Step {} Batch Loss {:.4f}'.format(
                            (batch), batch_loss))
                    else:
                        eval_loss = eval_step(inp, tar)
                        print('Step {} Eval Loss {:.4f}'.format(
                            (batch), eval_loss))
                    pbar.update(1)

            print('Epoch {} Loss {:.4f}'.format(
                        (epoch), batch_loss))
            print('Time taken for 1 Epoch: {} secs\n'.format(time.time() - start))
            optimizer._lr =  optimizer._lr * (args.decay_rate)**(epoch // 1)
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint \n")
        
