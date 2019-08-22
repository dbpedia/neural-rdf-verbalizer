"""
Initializes GAT trainer and trains the model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import tensorflow as tf
from tqdm import tqdm

from src.DataLoader import GetGATDataset
from src.models.GraphAttentionModel import TransGAT, GATModel
from src.utils.metrics import LossLayer
from src.utils.model_utils import CustomSchedule, _set_up_dirs
from src.utils.rogue import rouge_n


def _train_gat_rnn(args):
    # set up dirs
    (OUTPUT_DIR, EvalResultsFile,
     TestResults, log_file, log_dir) = _set_up_dirs(args)

    OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    (dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
     src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab, max_length_targ, dataset_size) = GetGATDataset(args)

    model = GATModel(args, src_vocab_size, tgt_vocab_size, tgt_vocab)

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
        model=model,
        optimizer=optimizer
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
        batch_loss = (loss / int(targ.shape[1]))
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

    total_loss = 0
    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        print('Learning Rate' + str(optimizer._lr) + ' Step ' + str(step))
        with tqdm(total=(38668 // args.batch_size)) as pbar:
            for (batch, (adj, nodes, edges, roles, targ)) in tqdm(enumerate(dataset)):
                start = time.time()
                step += 1
                if args.decay is not None:
                    optimizer._lr = learning_rate(tf.cast(step, dtype=tf.float32))

                if batch % args.eval_steps == 0:
                    eval_loss = eval_step(adj, nodes, roles, targ)
                    print('\n' + '---------------------------------------------------------------------' + '\n')
                    print('Epoch {} Batch {} Eval Loss {:.4f} '.format(epoch, batch,
                                                                       eval_loss.numpy()))
                    print('\n' + '---------------------------------------------------------------------' + '\n')
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


def _train_gat_trans(args):
    # set up dirs
    (OUTPUT_DIR, EvalResultsFile,
     TestResults, log_file, log_dir) = _set_up_dirs(args)

    OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    (dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
     src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab, max_length_targ, dataset_size) = GetGATDataset(args)

    # Load the eval src and tgt files for evaluation
    reference = open(args.eval_ref, 'r')
    eval_file = open(args.eval, 'r')

    model = TransGAT(args, src_vocab_size, src_vocab,
                     tgt_vocab_size, max_length_targ, tgt_vocab)
    loss_layer = LossLayer(tgt_vocab_size, 0.1)
    if args.decay is not None:
        learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)

    # Save model parameters for future use
    if os.path.isfile(log_dir + '/' + args.lang + '_model_params'):
        with open(log_dir + '/' + args.lang + '_model_params', 'rb') as fp:
            PARAMS = pickle.load(fp)
            print('Loaded Parameters..')
    else:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        PARAMS = {
            "args": args,
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "max_tgt_length": max_length_targ,
            "dataset_size": dataset_size,
            "step": 0
        }

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
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

    def eval_step(steps=None):
        model.trainable = False
        results = []
        ref_target = []
        eval_results = open(EvalResultsFile, 'w+')
        if steps is None:
            dev_set = eval_set
        else:
            dev_set = eval_set.take(steps)

        for (batch, (nodes, labels, node1, node2)) in tqdm(enumerate(dev_set)):
            predictions = model(nodes, labels, node1,
                                node2, targ=None, mask=None)
            pred = [(predictions['outputs'].numpy().tolist())]

            if args.sentencepiece == 'True':
                for i in range(len(pred[0])):
                    sentence = (tgt_vocab.DecodeIds(list(pred[0][i])))
                    sentence = sentence.partition("<start>")[2].partition("<end>")[0]
                    eval_results.write(sentence + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence)
            else:
                for i in pred:
                    sentences = tgt_vocab.sequences_to_texts(i)
                    sentence = [j.partition("<start>")[2].partition("<end>")[0] for j in sentences]
                    for w in sentence:
                        eval_results.write((w + '\n'))
                        ref_target.append(reference.readline())
                        results.append(w)

        rogue = (rouge_n(results, ref_target))
        score = 0
        eval_results.close()
        model.trainable = True

        return rogue, score

    # Eval function
    def test_step():
        model.trainable = False
        results = []
        ref_target = []
        eval_results = open(TestResults, 'w+')

        for (batch, (nodes, labels, node1, node2)) in tqdm(enumerate(test_set)):
            predictions = model(nodes, labels, node1,
                                node2, targ=None, mask=None)
            pred = [(predictions['outputs'].numpy().tolist())]
            if args.sentencepiece == 'True':
                for i in range(len(pred[0])):
                    sentence = (tgt_vocab.DecodeIds(list(pred[0][i])))
                    sentence = sentence.partition("<start>")[2].partition("<end>")[0]
                    eval_results.write(sentence + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence)
            else:
                for i in pred:
                    sentences = tgt_vocab.sequences_to_texts(i)
                    sentence = [j.partition("<start>")[2].partition("<end>")[0] for j in sentences]
                    for w in sentence:
                        eval_results.write((w + '\n'))
                        ref_target.append(reference.readline())
                        results.append(w)
        rogue = (rouge_n(results, ref_target))
        score = 0
        eval_results.close()
        model.trainable = True

        return rogue, score

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (nodes, labels,
                 node1, node2, targ)) in tqdm(enumerate(dataset.repeat(-1))):
        if PARAMS['step'] < steps:
            start = time.time()
            PARAMS['step'] += 1

            if args.decay is not None:
                optimizer._lr = learning_rate(tf.cast(PARAMS['step'], dtype=tf.float32))

            batch_loss, acc, ppl = train_step(nodes, labels, node1, node2, targ)
            if batch % 100 == 0:
                print('Step {} Learning Rate {:.4f} Train Loss {:.4f} '
                      'Accuracy {:.4f} Perplex {:.4f}'.format(PARAMS['step'],
                                                              optimizer._lr,
                                                              train_loss.result(),
                                                              acc.numpy(),
                                                              ppl.numpy()))
                print('Time {} \n'.format(time.time() - start))
            # log the training results
            tf.io.write_file(log_file,
                             f"Step {PARAMS['step']} Train Accuracy: {acc.numpy()}"
                             f" Loss: {train_loss.result()} Perplexity: {ppl.numpy()} \n")

            if batch % args.eval_steps == 0:
                rogue, score = eval_step(5)
                print('\n' + '---------------------------------------------------------------------' + '\n')
                print('Rogue {:.4f} BLEU {:.4f}'.format(rogue, score))
                print('\n' + '---------------------------------------------------------------------' + '\n')

            if batch % args.checkpoint == 0:
                print("Saving checkpoint \n")
                ckpt_save_path = ckpt_manager.save()
                with open(log_dir + '/' + args.lang + '_model_params', 'wb+') as fp:
                    pickle.dump(PARAMS, fp)

        else:
            break
    rogue, score = test_step()
    print('\n' + '---------------------------------------------------------------------' + '\n')
    print('Rogue {:.4f} BLEU {:.4f}'.format(rogue, score))
    print('\n' + '---------------------------------------------------------------------' + '\n')