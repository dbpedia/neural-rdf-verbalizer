""" Script to train the selected model
    Used to train a Multi-lingual model ( Teacher model )
    Loads individual student models and then
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import os
import argparse
from tqdm import tqdm
import time

from src.MultilingualDataLoader import ProcessMultilingualDataset
from src.arguments import get_args
from src.models import model_params, GraphAttentionModel, RNNModel
from src.utils.model_utils import CustomSchedule
from src.utils.Optimizers import LazyAdam
from src.models.GraphAttentionModel import TransGAT
from src.models.Transformer import Transformer
from src.utils.metrics import LossLayer
from inference import Inference
from src.utils.rogue import rouge_n
from nltk.translate.bleu_score import corpus_bleu

parser = argparse.ArgumentParser(description="Main Arguments")

# model paramteres

if __name__ == "__main__":
    args = get_args()
    global step

    if args.use_colab is None:
        output_file = 'results.txt'
        OUTPUT_DIR = 'ckpts/' + args.lang
        log_file = 'data/logs/' + args.lang + '_' + args.enc_type + '_' + str(args.emb_dim) + '.log'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    else:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/' + args.lang
        output_file = OUTPUT_DIR + '/results.txt'
        log_file = OUTPUT_DIR + '/logs/' + args.lang + '_' + args.enc_type + '_' + str(args.emb_dim) + '.txt'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    if args.enc_type == 'gat' and args.dec_type == 'rnn':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == 'rnn' and args.dec_type == 'rnn':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == 'transformer' and args.dec_type == 'transformer':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == 'gat' and args.dec_type == 'transformer':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type
        (dataset, eval_dataset, test_set, train_buffer_size, eval_buffer_size, BATCH_SIZE, steps_per_epoch,
         src_vocab, src_vocab_size, tgt_vocab, tgt_vocab_size , max_length, dataset_size) = ProcessMultilingualDataset(args)

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
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        step = 0

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

            for (batch, (nodes, labels, node1, node2)) in tqdm(enumerate(eval_dataset)):
                predictions = model(nodes, labels, node1,
                                    node2, targ=None, mask=None)
                pred = [(predictions['outputs'].numpy().tolist())]
                for i in range(len(pred[0])):
                    results.append(tgt_vocab.DecodeIds(list(pred[0][i])))

            rogue = (rouge_n(results, ref_target))
            score = corpus_bleu(ref_target, results)
            file.close()
            model.trainable = True

            return rogue, score


        for epoch in range(args.epochs):
            print('Learning Rate' + str(optimizer._lr) + ' Step ' + str(step))
            print(dataset_size)
            with tqdm(total=(int(dataset_size // args.batch_size))) as pbar:
                train_loss.reset_states()
                train_accuracy.reset_states()

                for (batch, (nodes, labels, node1, node2, targ)) in tqdm(enumerate(dataset)):
                    start = time.time()
                    step += 1
                    if args.decay is not None:
                        optimizer._lr = learning_rate(tf.cast(step, dtype=tf.float32))

                    if batch % args.eval_steps == 0:
                        rogue, score = eval_step()
                        print('\n' + '---------------------------------------------------------------------' + '\n')
                        print('Rogue {:.4f} BLEU {:.4f}'.format(rogue, score))
                        print('\n' + '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss, acc, ppl = train_step(nodes, labels, node1, node2, targ)
                        print('Epoch {} Batch {} Train Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                                                          train_loss.result(),
                                                                                                          acc.numpy(),
                                                                                                          ppl.numpy()))
                        # log the training results
                        tf.io.write_file(log_file,
                                         f'Epoch {epoch} Train Accuracy: {acc.numpy()} Loss: {train_loss.result()} Perplexity: {ppl.numpy()} \n')

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} \n'.format(time.time() - start))

                    pbar.update(1)