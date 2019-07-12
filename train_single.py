""" Script to train the selected model
    Used to train a single language model ( Teacher model ) 
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import time
import os
from tqdm import tqdm 

from src.data_loader import get_dataset, get_gat_dataset
from src.models import model_params, graph_attention_model, rnn_model
from src.utils.model_utils import create_transgat_masks
from src.utils.model_utils import CustomSchedule
from src.utils.optimizers import LazyAdam
from src.arguments import get_args
from src.models.graph_attention_model import TransGAT
from src.models.transformer import Transformer
from src.utils.metrics import LossLayer

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

if __name__ == "__main__":
    args = get_args()
    global step

    #set up dirs
    if args.use_colab is None:
        output_file = 'results.txt'
        OUTPUT_DIR = 'ckpts/'+args.lang
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    else:
        from google.colab import drive
        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/'+args.lang
        output_file = OUTPUT_DIR + '/results.txt'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    if args.enc_type == 'gat' and args.dec_type =='rnn':
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type
        (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
         vocab_tgt_size, vocab_nodes_size, vocab_edge_size, vocab_role_size,
         target_lang, max_length_targ) = get_gat_dataset(args)

        embedding = tf.keras.layers.Embedding(vocab_nodes_size, args.emb_dim)
        model = graph_attention_model.GATModel(args, vocab_nodes_size,
                                               vocab_role_size,vocab_tgt_size, target_lang)

        step = 0
        if args.decay is not None:
            learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
            
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        ckpt = tf.train.Checkpoint(
            model = model,
            optimizer = optimizer
        )
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            
        if args.epochs is not None:
            steps = args.epochs * steps_per_epoch
        else:
            steps = args.steps

        def train_step(adj, nodes, roles, targ):
            with tf.GradientTape() as tape:
                predictions, dec_hidden, loss = model(adj, nodes, roles, targ)
                reg_loss = tf.losses.get_regularization_loss()
                loss += reg_loss
            batch_loss =(loss / int(targ.shape[1]))
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        # Eval function
        def eval_step(adj, nodes, roles, targ):
            model.trainable = False
            predictions, dec_hidden, loss = model(adj, nodes, roles, targ)
            eval_loss = (loss / int(targ.shape[1]))
            model.trainable = True
            train_loss(eval_loss)
            eval_loss = train_loss.result()

            return eval_loss

        total_loss =0
        for epoch in range(args.epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            print('Learning Rate'+str(optimizer._lr)+' Step '+ str(step))
            with tqdm(total=(38668 // args.batch_size)) as pbar:
                for (batch, (adj, nodes, edges, roles, targ)) in tqdm(enumerate(dataset)):
                    start = time.time()
                    step += 1
                    if args.decay is not None:
                        optimizer._lr = learning_rate(tf.cast(step, dtype=tf.float32))
                  
                    if batch % args.eval_steps == 0:
                        eval_loss = eval_step(adj, nodes, roles, targ)
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                        print('Epoch {} Batch {} Eval Loss {:.4f} '.format(epoch, batch,
                                                                           eval_loss.numpy()))
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss = train_step(adj, nodes, roles, targ)
                        print('Epoch {} Batch {} Batch Loss {:.4f} '.format(epoch, batch,
                                                                           batch_loss.numpy()))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} \n'.format(time.time() - start))
                    pbar.update(1)

    elif args.enc_type == 'rnn' and args.dec_type =="rnn":
      
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type
        dataset, BUFFER_SIZE, BATCH_SIZE,\
        steps_per_epoch, vocab_inp_size, vocab_tgt_size, target_lang = get_dataset(args)

        step = 0

        if args.decay is not None:
            learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
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

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)

        def train_step(inp, targ, enc_hidden):

            with tf.GradientTape() as tape:
                predictions, dec_hidden, loss = model(inp, targ, enc_hidden)
                reg_loss = tf.losses.get_regularization_loss()
                loss += reg_loss

            batch_loss = (loss / int(targ.shape[1]))
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        def eval_step(inp, trg, enc_hidden):
            model.trainable = False

            predictions, dec_hidden, eval_loss = model(inp, trg, enc_hidden)
            eval_loss = (eval_loss / int(targ.shape[1]))
            model.trainable = True

            return eval_loss

        for epoch in range(args.epochs):
            print('Learning Rate'+str(optimizer._lr)+' Step '+ str(step))

            with tqdm(total=(38668 // args.batch_size)) as pbar:
                for (batch, (inp, targ)) in tqdm(enumerate(dataset)):
                    start = time.time()
                    step += 1
                    if args.decay is not None:
                        optimizer._lr = learning_rate(tf.cast(step, dtype=tf.float32))

                    if batch % args.eval_steps == 0:
                        eval_loss = eval_step(inp, targ, enc_hidden)
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                        print('Epoch {} Batch {} Eval Loss {:.4f} '.format(epoch, batch,
                                                                           eval_loss.numpy()))
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss = train_step(inp, targ, enc_hidden)
                        print('Epoch {} Batch {} Batch Loss {:.4f} '.format(epoch, batch,
                                                                           batch_loss.numpy()))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} '.format(time.time() - start))
                    pbar.update(1)
            if args.decay is not None:
                optimizer._lr = optimizer._lr * args.decay_rate ** (batch // 1)

    elif args.enc_type == 'transformer' and args.dec_type =="transformer":
      
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type
        dataset, BUFFER_SIZE, BATCH_SIZE,\
        steps_per_epoch, vocab_size, lang= get_dataset(args)
        num_layers = args.enc_layers
        num_heads = args.num_heads
        d_model = args.emb_dim
        dff = args.hidden_size
        dropout_rate = args.dropout
        if args.epochs is None :
            epochs = args.steps // steps_per_epoch
        else:
            epochs = args.epochs

        step = 0

        if args.decay is not None:
            learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
            optimizer = LazyAdam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        else:
            optimizer = LazyAdam(learning_rate=args.learning_rate,
                                 beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        
        model = Transformer(args, vocab_size)
        loss_layer = LossLayer(vocab_size, 0.1)

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
            with tf.GradientTape() as tape:
                predictions = model(inp, tar, training=model.trainable)
                predictions = model.metric_layer([predictions, tar])
                loss = loss_layer([predictions, tar])
                reg_loss = tf.losses.get_regularization_loss()
                loss += reg_loss

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            acc = model.metrics[0].result()
            ppl = model.metrics[-1].result()

            return loss, acc, ppl

        def eval_step(inp, tar):
            model.trainable = False
            predictions = model(inp, tar, training=model.trainable)
            predictions = model.metric_layer([predictions, tar])
            loss = loss_layer([predictions, tar])
            acc = model.metrics[0].result()
            ppl = model.metrics[-1].result()
            model.trainable = True

            return loss, acc, ppl

        for epoch in range(epochs):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()
            print('Learning Rate'+str(optimizer.lr)+' Step '+ str(step))
            with tqdm(total=(38668 // args.batch_size)) as pbar:
                for (batch, (inp, tar)) in tqdm(enumerate(dataset)):
                    step += 1
                    if args.decay is not None:
                        optimizer.lr = learning_rate(tf.cast(step, dtype=tf.float32))

                    if (batch % args.eval_steps == 0):
                        eval_loss, acc, ppl = eval_step(inp, tar)
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                        print('Epoch {} Batch {} Eval Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                                                          eval_loss.numpy(),
                                                                                                          acc.numpy(),
                                                                                                          ppl.numpy()))
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss, acc, ppl = train_step(inp, tar)
                        print('Epoch {} Batch {} Train Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                                                          batch_loss.numpy(),
                                                                                                          acc.numpy(),
                                                                                                          ppl.numpy()))
                    pbar.update(1)

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        (epoch), train_loss.result(), train_accuracy.result()))
            print('Time taken for 1 Epoch: {} secs\n'.format(time.time() - start))
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint \n")

    elif ((args.enc_type == "gat")and(args.dec_type == "transformer")):
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type
        (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
         vocab_tgt_size, vocab_nodes_size, vocab_edge_size, vocab_role_size,
         target_lang, max_length_targ) = get_gat_dataset(args)
        model = TransGAT(args, vocab_nodes_size, vocab_role_size,
                                            vocab_tgt_size, target_lang)
        loss_layer = LossLayer(vocab_tgt_size, 0.1)
        if args.decay is not None:
            learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        step =0

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        ckpt = tf.train.Checkpoint(
            model=model
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        if args.epochs is not None:
            steps = args.epochs * steps_per_epoch
        else:
            steps = args.steps

        def train_step(adj, nodes, roles, targ):
            with tf.GradientTape() as tape:
                mask = create_transgat_masks(targ)
                predictions = model(adj, nodes, roles, targ, mask)
                predictions = model.metric_layer([predictions, targ])
                batch_loss = loss_layer([predictions, targ])

            gradients = tape.gradient(batch_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc = model.metrics[0].result()
            ppl = model.metrics[-1].result()

            return batch_loss, acc, ppl

         # Eval function
        def eval_step(adj, nodes, roles, targ):
            model.trainable = False
            mask = create_transgat_masks(targ)
            predictions = model(adj, nodes, roles, targ, mask)
            predictions = model.metric_layer([predictions, targ])
            eval_loss = loss_layer([predictions, targ])
            acc = model.metrics[0].result()
            ppl = model.metrics[-1].result()
            model.trainable = True

            return eval_loss, acc, ppl

        for epoch in range(args.epochs):
            print('Learning Rate'+str(optimizer._lr)+' Step '+ str(step))
            with tqdm(total=(38668 // args.batch_size)) as pbar:
                train_loss.reset_states()
                train_accuracy.reset_states()

                for (batch, (adj, nodes, edges, roles, targ)) in tqdm(enumerate(dataset)):
                    start = time.time()
                    step +=1
                    if args.decay is not None:
                        optimizer._lr = learning_rate(tf.cast(step, dtype=tf.float32))

                    if batch % args.eval_steps == 0:
                        eval_loss, acc, ppl = eval_step(adj, nodes, roles, targ)
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                        print('Epoch {} Batch {} Eval Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                 eval_loss.numpy(), acc.numpy(), ppl.numpy()))
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss, acc, ppl = train_step(adj, nodes, roles, targ)
                        print('Epoch {} Batch {} Train Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                  batch_loss.numpy(), acc.numpy(), ppl.numpy()))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} \n'.format(time.time() - start))

                    pbar.update(1)