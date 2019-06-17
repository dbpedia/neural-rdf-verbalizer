"""     Defines the mgraph attention encoder model
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
tf.enable_eager_execution()
from src.layers.encoders import GraphEncoder
from src.layers.decoders import RNNDecoder
from src.utils.model_utils import loss_function

class GATModel (tf.keras.Model):
    """
    Model that uses Graph Attention encoder and RNN decoder (for now)
    """
    def __init__(self, args, vocab_tgt_size, target_lang):
        super(GATModel, self).__init__()
        self.encoder = GraphEncoder(args)
        self.decoder = RNNDecoder(vocab_tgt_size, args.emb_dim, args.enc_units, args.batch_size)
        self.vocab_tgt_size = vocab_tgt_size 
        self.target_lang = target_lang
        self.args = args
        self.loss_object = tf.keras.losses.sparse_categorical_crossentropy

    def __call__(self, adj, nodes, targ):
        """
        Puts the tensors through encoders and decoders
        :param adj: Adjacency matrices of input example
        :type adj: tf.tensor
        :param nodes: node features
        :type nodes: tf.tensor
        :param targ: target sequences
        :type targ: tf.tensor
        :return: output probability distribution
        :rtype: tf.tensor
        """
        enc_output, enc_hidden = self.encoder(nodes, adj, self.encoder.trainable)
        dec_input=tf.expand_dims([self.target_lang.word_index['<start>']] * self.args.batch_size, 1)
        loss = 0
        
        for t in range(1, targ.shape[1]):
            # pass encoder output to decoder
            predictions, dec_hidden, _ = self.decoder(dec_input, enc_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions, self.loss_object)

            #using teacher forcing 
            dec_input = tf.expand_dims(targ[:, t], 1)

        return predictions, dec_hidden, loss

        
        
        

