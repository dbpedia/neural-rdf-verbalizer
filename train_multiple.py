""" Script to train the selected model
    Used to train a Multi-lingual model ( Teacher model )
    Loads individual student models and then
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import tensorflow as tf
from tqdm import tqdm

from src.MultilingualDataLoader import ProcessMultilingualDataset
from src.arguments import get_args
from src.models.GraphAttentionModel import TransGAT
from src.utils.metrics import LossLayer
from src.utils.model_utils import CustomSchedule, _set_up_dirs
from src.utils.rogue import rouge_n

# model paramteres

if __name__ == "__main__":
  args = get_args()
  global step

  # set up dirs
  (OUTPUT_DIR, EvalResultsFile,
   TestResults, log_file, log_dir) = _set_up_dirs(args)

  if args.enc_type == 'gat' and args.dec_type == 'transformer':
    OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type
    (dataset, src_vocab, src_vocab_size, tgt_vocab,
     tgt_vocab_size, MULTI_BUFFER_SIZE, steps_per_epoch, MaxSeqSize) = ProcessMultilingualDataset(args)

    # Load the eval src and tgt files for evaluation
    reference = open(args.eval_ref, 'r')
    eval_file = open(args.eval, 'r')

    model = TransGAT(args, src_vocab_size, src_vocab,
                     tgt_vocab_size, MaxSeqSize, tgt_vocab)
    loss_layer = LossLayer(tgt_vocab_size, 0.1)
    if args.decay is not None:
      learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                         epsilon=1e-9)
    else:
      optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=0.9, beta2=0.98,
                                         epsilon=1e-9)
    step = 0

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
        "step": 0
      }

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
    def eval_step(steps):
      model.trainable = False
      results = []
      ref_target = []
      eval_results = open(EvalResultsFile, 'w+')

      for (batch, (nodes, labels,
                   node1, node2, target)) in tqdm(enumerate(
        dataset['eval_set'].take(steps))):
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
      eval_results.close()
      model.trainable = True

      return rogue


    # Eval function
    def test_step():
      model.trainable = False
      results = []
      ref_target = []
      eval_results = open(TestResults, 'w+')

      for (batch, (nodes, labels, node1, node2)) in tqdm(enumerate(
              dataset['test_set'])):
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
                 node1, node2, targ)) in tqdm(enumerate(
      dataset['train_set'].repeat(-1))):
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
          rogue = eval_step(5)
          print('\n' + '---------------------------------------------------------------------' + '\n')
          print('Rogue {:.4f}'.format(rogue))
          print('\n' + '---------------------------------------------------------------------' + '\n')

        if batch % args.checkpoint == 0:
          ckpt_save_path = ckpt_manager.save()
          print("Saving checkpoint \n")
          with open(log_dir + '/' + args.lang + '_model_params', 'wb+') as fp:
            pickle.dump(PARAMS, fp)

      else:
        break
    rogue, score = test_step()
    print('\n' + '---------------------------------------------------------------------' + '\n')
    print('Rogue {:.4f} BLEU {:.4f}'.format(rogue, score))
    print('\n' + '---------------------------------------------------------------------' + '\n')
