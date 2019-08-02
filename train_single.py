""" Script to train the selected model
    Used to train a single language model ( Teacher model ) 
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import time
import os
import logging
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from src.DataLoader import GetDataset, GatGATdataset
from src.models import model_params, GraphAttentionModel, RNNModel
from src.utils.model_utils import CustomSchedule
from src.utils.Optimizers import LazyAdam
from src.arguments import get_args
from src.models.GraphAttentionModel import TransGAT
from src.models.Transformer import Transformer
from src.utils.metrics import LossLayer
from inference import Inference
from src.utils.rogue import rouge_n

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
        log_file = 'data/logs/'+args.lang+'_'+args.enc_type+'_'+str(args.emb_dim)+'.log'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    else:
        from google.colab import drive
        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/'+args.lang
        output_file = OUTPUT_DIR + '/results.txt'
        log_file = OUTPUT_DIR+'/logs/' + args.lang + '_' + args.enc_type + '_' + str(args.emb_dim) + '.txt'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    if args.enc_type == 'gat' and args.dec_type =='rnn':
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type

        (dataset, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
         vocab_tgt_size, vocab_nodes_size, vocab_edge_size, vocab_role_size,
         target_lang, max_length_targ) = GatGATdataset(args)

        embedding = tf.keras.layers.Embedding(vocab_nodes_size, args.emb_dim)
        model = GraphAttentionModel.GATModel(args, vocab_nodes_size,
                                             vocab_role_size, vocab_tgt_size, target_lang)

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
                        # log the training results
                        tf.io.write_file(log_file, "Epoch {}".format(epoch))
                        tf.io.write_file(log_file, "Train Loss: {}".format(batch_loss))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} \n'.format(time.time() - start))
                    pbar.update(1)

    elif args.enc_type == 'rnn' and args.dec_type =="rnn":
      
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type

        dataset, BUFFER_SIZE, BATCH_SIZE,\
        steps_per_epoch, vocab_inp_size, vocab_tgt_size, target_lang = GetDataset(args)

        step = 0

        if args.decay is not None:
            learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        model = RNNModel.RNNModel(vocab_inp_size, vocab_tgt_size, target_lang, args)
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
                        # log the training results
                        tf.io.write_file(log_file, "Epoch {}".format(epoch))
                        tf.io.write_file(log_file, "Train Loss: {}".format(batch_loss))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} '.format(time.time() - start))
                    pbar.update(1)
            if args.decay is not None:
                optimizer._lr = optimizer._lr * args.decay_rate ** (batch // 1)

    elif args.enc_type == 'transformer' and args.dec_type =="transformer":
      
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type

        dataset, BUFFER_SIZE, BATCH_SIZE, \
        steps_per_epoch, src_vocab_size, lang, dataset_size= GetDataset(args)
        ref_sentence = []
        reference = open(args.eval_ref, 'r')
        for i, line in enumerate(reference):
            if (i < (args.num_eval_lines)):
                ref_sentence.append(line)
        eval_file = open(args.eval, 'r')
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
        
        model = Transformer(args, src_vocab_size)
        loss_layer = LossLayer(src_vocab_size, 0.1)

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
            file = open(output_file, 'w+')
            verbalised_triples = []
            for i, line in enumerate(eval_file):
                if i < args.num_eval_lines:
                    print(line)
                    result = Inference(args, line, model, lang)
                    file.write(result + '\n')
                    verbalised_triples.append(result)
                    print(str(i) + ' ' + result)
            rogue = (rouge_n(verbalised_triples, ref_sentence))
            # score = corpus_bleu(ref_sentence, verbalised_triples)
            file.close()
            model.trainable = True

            return rogue

        for epoch in range(epochs):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()
            print('Learning Rate'+str(optimizer.lr)+' Step '+ str(step))
            with tqdm(total=(dataset_size // args.batch_size)) as pbar:
                for (batch, (inp, tar)) in tqdm(enumerate(dataset)):
                    step += 1
                    if args.decay is not None:
                        optimizer.lr = learning_rate(tf.cast(step, dtype=tf.float32))

                    if (batch % args.eval_steps == 0):
                        eval_loss = eval_step(inp, tar)
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                        print('Epoch {} Batch {} Rouge {:.4f}'.format(epoch, batch, eval_loss   ))
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss, acc, ppl = train_step(inp, tar)
                        print('Epoch {} Batch {} Train Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                                                          batch_loss.numpy(),
                                                                                                          acc.numpy(),
                                                                                                          ppl.numpy()))
                        # log the training results
                        tf.io.write_file(log_file, "Epoch {}".format(epoch))
                        tf.io.write_file(log_file, "Train Accuracy: {}, Loss: {}, Perplexity{}".format((acc),
                                                                                         train_loss, ppl))

                    pbar.update(1)

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        (epoch), train_loss.result(), train_accuracy.result()))
            print('Time taken for 1 Epoch: {} secs\n'.format(time.time() - start))
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint \n")

    elif ((args.enc_type == "gat")and(args.dec_type == "transformer")):
        OUTPUT_DIR += '/' + args.enc_type+'_'+args.dec_type

        (dataset, eval_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
         src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab, max_length_targ, dataset_size) = GatGATdataset(args)

        # Load the eval src and tgt files for evaluation
        ref_source = []
        ref_target = []
        reference = open(args.eval_ref, 'r')
        eval_file = open(args.eval, 'r')
        for i, (eval_src, eval_tgt) in enumerate(zip(eval_file, reference)):
            if i < args.num_eval_lines:
                ref_source.append(eval_src)
                ref_target.append(eval_tgt)
        reference.close()
        eval_file.close()

        model = TransGAT(args, src_vocab_size, src_vocab,
                         tgt_vocab_size, tgt_vocab)
        loss_layer = LossLayer(tgt_vocab_size, 0.1)
        if args.decay is not None:
            learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1=0.9, beta2=0.98,
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

        def train_step(nodes, labels, node1, node2, targ):
            with tf.GradientTape() as tape:
                predictions = model(nodes, labels, node1, node2, targ, None)
                predictions = model.metric_layer([predictions, targ])
                batch_loss = loss_layer([predictions, targ])

            gradients = tape.gradient(batch_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc = model.metrics[0].result()
            ppl = model.metrics[-1].result()
            batch_loss = train_loss(batch_loss)

            return batch_loss, acc, ppl

         # Eval function
        def eval_step():
            model.trainable = False
            results = []
            file = open(output_file, 'w+')

            for (batch, (nodes, labels, node1, node2)) in tqdm(enumerate(eval_set)):
                predictions = model(nodes, labels, node1,
                                    node2, targ=None, mask=None)
                pred = [(predictions['outputs'].numpy().tolist())]
                for i in range(len(pred[0])):
                    sentence = (tgt_vocab.DecodeIds(list(pred[0][i])))
                    sentence = sentence.partition("start")[2].partition("end")[0]
                    print(sentence+'\n')
                    results.append(sentence)

            rogue = (rouge_n(results, ref_target))
            score = corpus_bleu(ref_target, results)
            file.close()
            model.trainable = True

            return rogue, score

        for epoch in range(args.epochs):
            print('Learning Rate'+str(optimizer._lr)+' Step '+ str(step))
            print(dataset_size)
            with tqdm(total=(dataset_size // args.batch_size)) as pbar:
                train_loss.reset_states()
                train_accuracy.reset_states()

                for (batch, (nodes, labels, node1, node2, targ)) in tqdm(enumerate(dataset)):
                    start = time.time()
                    step +=1
                    if args.decay is not None:
                        optimizer._lr = learning_rate(tf.cast(step, dtype=tf.float32))

                    if batch % args.eval_steps == 0:
                        rogue, score = eval_step()
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                        print('Rogue {:.4f} BLEU {:.4f}'.format(rogue, score))
                        print('\n'+ '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss, acc, ppl = train_step(nodes, labels, node1, node2, targ)
                        print('Epoch {} Batch {} Train Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                  train_loss.result(), acc.numpy(), ppl.numpy()))
                        # log the training results
                        tf.io.write_file(log_file,
                                         f'Epoch {epoch} Train Accuracy: {acc.numpy()} Loss: {train_loss.result()} Perplexity: {ppl.numpy()} \n')

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} \n'.format(time.time() - start))

                    pbar.update(1)