from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from tqdm import tqdm

from src.DataLoader import GetDataset
from src.models import RNNModel
from src.utils.model_utils import CustomSchedule, _set_up_dirs


def _train_rnn(args):
  # set up dirs
  (OUTPUT_DIR, EvalResultsFile,
   TestResults, log_file, log_dir) = _set_up_dirs(args)

  OUTPUT_DIR += '/{}_{}'.format(args.enc_type, args.dec_type)

  dataset, BUFFER_SIZE, BATCH_SIZE, \
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
    model=model,
    optimizer=optimizer
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
    print('Learning Rate' + str(optimizer._lr) + ' Step ' + str(step))

    with tqdm(total=(38668 // args.batch_size)) as pbar:
      for (batch, (inp, targ)) in tqdm(enumerate(dataset)):
        start = time.time()
        step += 1
        if args.decay is not None:
          optimizer._lr = learning_rate(tf.cast(step, dtype=tf.float32))

        if batch % args.eval_steps == 0:
          eval_loss = eval_step(inp, targ, enc_hidden)
          print('\n' + '---------------------------------------------------------------------' + '\n')
          print('Epoch {} Batch {} Eval Loss {:.4f} '.format(epoch, batch,
                                                             eval_loss.numpy()))
          print('\n' + '---------------------------------------------------------------------' + '\n')
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
