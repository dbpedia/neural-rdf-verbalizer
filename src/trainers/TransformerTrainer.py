"""
Initializes Pure Transformer trainer and trains the model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import tensorflow as tf
from tqdm import tqdm

from src.DataLoader import GetDataset
from src.models.Transformer import Transformer
from src.utils.Optimizers import LazyAdam
from src.utils.metrics import LossLayer
from src.utils.model_utils import CustomSchedule, _set_up_dirs
from src.utils.rogue import rouge_n


def _train_transformer(args):
  # set up dirs
  (OUTPUT_DIR, EvalResultsFile,
   TestResults, log_file, log_dir) = _set_up_dirs(args)

  OUTPUT_DIR += '/{}_{}'.format(args.enc_type, args.dec_type)

  dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, \
  steps_per_epoch, src_vocab_size, vocab, dataset_size, max_seq_len = GetDataset(args)
  reference = open(args.eval_ref, 'r')

  if args.epochs is not None:
    steps = args.epochs * steps_per_epoch
  else:
    steps = args.steps

  # Save model parameters for future use
  if os.path.isfile('{}/{}_{}_params'.format(log_dir, args.lang, args.model)):
    with open('{}/{}_{}_params'.format(log_dir, args.lang, args.model), 'rb') as fp:
      PARAMS = pickle.load(fp)
      print('Loaded Parameters..')
  else:
    if not os.path.isdir(log_dir):
      os.makedirs(log_dir)
    PARAMS = {
      "args": args,
      "vocab_size": src_vocab_size,
      "dataset_size": dataset_size,
      "max_tgt_length": max_seq_len,
      "step": 0
    }

  if args.decay is not None:
    learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
    optimizer = LazyAdam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  else:
    optimizer = LazyAdam(learning_rate=args.learning_rate,
                         beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

  model = Transformer(args, src_vocab_size)
  loss_layer = LossLayer(src_vocab_size, 0.1)

  ckpt = tf.train.Checkpoint(
    model=model,
    optimizer=optimizer
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

  def eval_step(steps=None):
    model.trainable = False
    results = []
    ref_target = []
    eval_results = open(EvalResultsFile, 'w+')
    if steps is None:
      dev_set = eval_set
    else:
      dev_set = eval_set.take(steps)
    for (batch, (inp, tar)) in tqdm(enumerate(dev_set)):
      predictions = model(inp, targets=None, training=model.trainable)
      pred = [(predictions['outputs'].numpy().tolist())]

      if args.sentencepiece == 'True':
        for i in range(len(pred[0])):
          sentence = (vocab.DecodeIds(list(pred[0][i])))
          sentence = sentence.partition("<start>")[2].partition("<end>")[0]
          eval_results.write(sentence + '\n')
          ref_target.append(reference.readline())
          results.append(sentence)
      else:
        for i in pred:
          sentences = vocab.sequences_to_texts(i)
          sentence = [j.partition("start")[2].partition("end")[0] for j in sentences]
          for w in sentence:
            eval_results.write((w + '\n'))
            ref_target.append(reference.readline())
            results.append(w)

    rogue = (rouge_n(results, ref_target))
    score = 0
    eval_results.close()
    model.trainable = True

    return rogue, score

  def test_step():
    model.trainable = False
    results = []
    ref_target = []
    eval_results = open(TestResults, 'w+')
    for (batch, (inp)) in tqdm(enumerate(test_set)):
      predictions = model(inp, targets=None, training=model.trainable)
      pred = [(predictions['outputs'].numpy().tolist())]

      if args.sentencepiece == 'True':
        for i in range(len(pred[0])):
          sentence = (vocab.DecodeIds(list(pred[0][i])))
          sentence = sentence.partition("<start>")[2].partition("<end>")[0]
          eval_results.write(sentence + '\n')
          ref_target.append(reference.readline())
          results.append(sentence)
      else:
        for i in pred:
          sentences = vocab.sequences_to_texts(i)
          sentence = [j.partition("start")[2].partition("end")[0] for j in sentences]
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

  for (batch, (inp, tgt)) in tqdm(enumerate(dataset.repeat(-1))):
    if PARAMS['step'] < steps:
      start = time.time()
      PARAMS['step'] += 1

      if args.decay is not None:
        optimizer._lr = learning_rate(tf.cast(PARAMS['step'], dtype=tf.float32))

      batch_loss, acc, ppl = train_step(inp, tgt)
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
        with open(log_dir + '/' + args.lang + '_' + args.model + '_params', 'wb+') as fp:
          pickle.dump(PARAMS, fp)

    else:
      break
  rogue, score = test_step()
  print('\n' + '---------------------------------------------------------------------' + '\n')
  print('Rogue {:.4f} BLEU {:.4f}'.format(rogue, score))
  print('\n' + '---------------------------------------------------------------------' + '\n')
