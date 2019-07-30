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

from src.multi_data_loader import process_gat_multidataset
from src.arguments import get_args
from src.models import model_params, graph_attention_model, rnn_model
from src.utils.model_utils import CustomSchedule
from src.utils.optimizers import LazyAdam
from src.models.graph_attention_model import TransGAT
from src.models.transformer import Transformer
from src.utils.metrics import LossLayer
from inference import inf
from src.utils.rogue import rouge_n
from src.utils.model_utils import convert

parser = argparse.ArgumentParser(description="Main Arguments")

# model paramteres

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

    if args.enc_type == 'gat' and args.dec_type == 'rnn':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == 'rnn' and args.dec_type == 'rnn':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == 'transformer' and args.dec_type == 'transformer':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == 'gat' and args.dec_type == 'transformer':
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type
        OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type
        (dataset, eval_dataset, train_buffer_size, eval_buffer_size, BATCH_SIZE, steps_per_epoch,
         vocab_size, vocab, max_length, dataset_size) = process_gat_multidataset(args)
        ref_sentence = []
        reference = open(args.eval_ref, 'r')
        for i, line in enumerate(reference):
            if (i < (args.num_eval_lines)):
                ref_sentence.append(line)
        eval_file = open(args.eval, 'r')

        model = TransGAT(args, vocab_size, vocab)
        loss_layer = LossLayer(vocab_size, 0.1)
        if args.decay is not None:
            learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                               epsilon=1e-9)
        else:
            optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
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
            batch = eval_dataset.take(1)
            print(batch[0].shape)
            exit(0)
            file = open(output_file, 'w+')
            verbalised_triples = []
            for i, line in enumerate(eval_file):
                if i < args.num_eval_lines:
                    result = inf(args, line, model, vocab)
                    file.write(result + '\n')
                    verbalised_triples.append(result)
                    print(str(i) + ' ' + result)
            rogue = (rouge_n(verbalised_triples, ref_sentence))
            # score = corpus_bleu(ref_sentence, verbalised_triples)
            file.close()
            model.trainable = True

            return rogue


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
                        #rogue = eval_step()
                        print('\n' + '---------------------------------------------------------------------' + '\n')
                        #print('Rogue {:.4f}'.format(rogue))
                        print('\n' + '---------------------------------------------------------------------' + '\n')
                    else:
                        batch_loss, acc, ppl = train_step(nodes, labels, node1, node2, targ)
                        print('Epoch {} Batch {} Train Loss {:.4f} Accuracy {:.4f} Perplex {:.4f}'.format(epoch, batch,
                                                                                                          train_loss.result(),
                                                                                                          acc.numpy(),
                                                                                                          ppl.numpy()))

                    if batch % args.checkpoint == 0:
                        ckpt_save_path = ckpt_manager.save()
                        print("Saving checkpoint \n")
                    print('Time {} \n'.format(time.time() - start))

                    pbar.update(1)