"""     Defines the mgraph attention encoder model
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from src.layers.encoders import GraphEncoder
from src.layers.decoders import RNNDecoder
from src.utils.model_utils import loss_function
from src.models.transformer import DecoderStack
from src.utils import transformer_utils
from src.layers import embedding_layer

class GATModel (tf.keras.Model):
    """
    Model that uses Graph Attention encoder and RNN decoder (for now)
    """
    def __init__(self, args, node_vocab_size, role_vocab_size, vocab_tgt_size, target_lang):
        super(GATModel, self).__init__()
        self.encoder = GraphEncoder(args.enc_layers, args.emb_dim, args.num_heads,
                                    args.hidden_size, node_vocab_size, role_vocab_size,
                                    reg_scale=args.reg_scale, rate=args.dropout)
        self.decoder = RNNDecoder(vocab_tgt_size, args.emb_dim, args.enc_units, args.batch_size)
        self.vocab_tgt_size = vocab_tgt_size
        self.num_heads = args.num_heads
        self.target_lang = target_lang
        self.args = args
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.hidden = tf.keras.layers.Dense(args.hidden_size)

    def __call__(self, adj, nodes, roles, targ):
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
        enc_output = self.encoder(nodes, adj, roles, self.num_heads, self.encoder.trainable, None)
        batch = enc_output.shape[0]
        self.enc_output_hidden = tf.reshape(enc_output, shape=[batch, -1])
        enc_hidden = self.hidden(self.enc_output_hidden)

        dec_input=tf.expand_dims([self.target_lang.word_index['<start>']] * self.args.batch_size, 1)
        loss = 0
        
        for t in range(1, targ.shape[1]):
            # pass encoder output to decoder
            predictions, dec_hidden, _ = self.decoder(dec_input, enc_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions, self.loss_object)

            #using teacher forcing 
            dec_input = tf.expand_dims(targ[:, t], 1)

        return predictions, dec_hidden, loss


class TransGAT(tf.keras.Model):
    """
    Model that uses Graph Attention encoder and RNN decoder (for now)
    """
    def __init__(self, args, node_vocab_size, role_vocab_size, vocab_tgt_size, target_lang):
        super(TransGAT, self).__init__()
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.emb_node_layer = embedding_layer.EmbeddingSharedWeights(
            node_vocab_size, args.hidden_size)
        self.emb_role_layer = embedding_layer.EmbeddingSharedWeights(
            role_vocab_size, args.hidden_size)
        self.emb_tgt_layer = embedding_layer.EmbeddingSharedWeights(
            vocab_tgt_size, args.hidden_size)

        self.encoder = GraphEncoder(args.enc_layers, args.emb_dim, args.num_heads,
                                    args.hidden_size, node_vocab_size, role_vocab_size,
                                    reg_scale= args.reg_scale, rate=args.dropout)
        self.decoder_stack = DecoderStack(args)
        self.vocab_tgt_size = vocab_tgt_size
        self.target_lang = target_lang
        self.args = args
        self.final_layer = tf.keras.layers.Dense(vocab_tgt_size)
        self.num_heads = args.num_heads

    def __call__(self, adj, nodes, roles, targ, mask):
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
        node_tensor = self.emb_node_layer(nodes)
        role_tensor = self.emb_role_layer(roles)
        decoder_inputs = self.emb_tgt_layer(targ)
        node_tensor = tf.cast(node_tensor, tf.float32)
        role_tensor = tf.cast(role_tensor, tf.float32)
        decoder_inputs = tf.cast(decoder_inputs, tf.float32)

        enc_output = self.encoder(node_tensor, adj, role_tensor,
                                  self.num_heads, self.encoder.trainable,None)
        attention_bias = transformer_utils.get_padding_bias(nodes)
        attention_bias = tf.cast(attention_bias, tf.float32)
        with tf.name_scope("shift_targets"):
            # Shift targets to the right, and remove the last element
            decoder_inputs = tf.pad(decoder_inputs,
                                    [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        attention_bias = tf.cast(attention_bias, tf.float32)
        with tf.name_scope("add_pos_encoding"):
            length = tf.shape(decoder_inputs)[1]
            pos_encoding = transformer_utils.get_position_encoding(
                length, self.args.hidden_size)
            pos_encoding = tf.cast(pos_encoding, tf.float32)
            decoder_inputs += pos_encoding
        if self.trainable:
            decoder_inputs = tf.nn.dropout(
                decoder_inputs, rate=self.args.dropout)

            # Run values
        decoder_self_attention_bias = transformer_utils.get_decoder_self_attention_bias(
            length, dtype=tf.float32)
        outputs = self.decoder_stack(
            decoder_inputs,
            enc_output,
            decoder_self_attention_bias,
            attention_bias,
            training=self.trainable)
        predictions = self.final_layer(outputs)

        return predictions