from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from src.layers import attention_layer
from src.layers import embedding_layer
from src.layers import ffn_layer
from src.utils import transformer_utils
from arguments import get_args

class LayerNormalization(tf.keras.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.scale = self.add_weight(
            "layer_norm_scale",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.ones_initializer(),
            experimental_autocast=False)
        self.bias = self.add_weight(
            "layer_norm_bias",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.zeros_initializer(),
            experimental_autocast=False)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
        }

    def call(self, x, epsilon=1e-6):
        input_dtype = x.dtype
        if input_dtype == tf.float16:
            x = tf.cast(x, tf.float32)
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)

        return tf.cast(norm_x * self.scale + self.bias, input_dtype)

class PrePostProcessingWrapper(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, args):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.args = args
        self.postprocess_dropout = args.dropout
        self.layer_norm = LayerNormalization(args.hidden_size)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]

        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)

        return x + y

class Transformer(tf.keras.Model):
    """Transformer model with Keras.
      Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf
      The Transformer model consists of an encoder and decoder. The input is an int
      sequence (or a batch of sequences). The encoder produces a continuous
      representation, and the decoder uses the encoder output to generate
      probabilities for the output sequence.
    """
    def __init__(self, args, vocab_tgt_size, name=None):
        super(Transformer, self).__init__(name=name)
        self.args = args
        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            vocab_tgt_size, args.hidden_size
        )
        self.encoder_stack = EncoderStack(args)
        self.decoder_stack = DecoderStack(args)

    def get_config(self):
        return {
            "args":self.args,
        }

    def call(self, inputs, targets, training):
        with tf.name_scope("Transformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = transformer_utils.get_padding_bias(inputs)
            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, attention_bias, training)
            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            if targets is None:
                return self.predict(encoder_outputs, attention_bias, training)
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias, training)
                return logits

    def encode(self, inputs, attention_bias, training):
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.embedding_softmax_layer(inputs)
            embedded_inputs = tf.cast(embedded_inputs, tf.float32)
            inputs_padding = transformer_utils.get_padding(inputs)
            attention_bias = tf.cast(attention_bias, tf.float32)

        with tf.name_scope("add_positional_encoding"):
            length = tf.shape(embedded_inputs)[1]
            pos_encoding = transformer_utils.get_position_encoding(
                length, self.args.hidden_size)
            pos_encoding = tf.cast(pos_encoding, tf.float32)
            encoder_inputs = embedded_inputs + pos_encoding

        if training:
            encoder_inputs = tf.nn.dropout(
                encoder_inputs, rate=self.args.dropout)

        return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding, training=training)

    def decode(self, targets, encoder_outputs, attention_bias, training):
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = tf.cast(decoder_inputs, tf.float32)
            attention_bias = tf.cast(attention_bias, tf.float32)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs,
                                        [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = transformer_utils.get_position_encoding(
                    length, self.args.hidden_size)
                pos_encoding = tf.cast(pos_encoding, tf.float32)
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.args.dropout)

                # Run values
            decoder_self_attention_bias = transformer_utils.get_decoder_self_attention_bias(
                length, dtype=tf.float32)
            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training)
            logits = self.embedding_softmax_layer(outputs, mode="linear")
            logits = tf.cast(logits, tf.float32)

            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = transformer_utils.get_position_encoding(
            max_decode_length + 1, self.args.hidden_size)
        decoder_self_attention_bias = transformer_utils.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            Args:
              ids: Current decoded sequences. int tensor with shape [batch_size *
                beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.
            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                cache.get("encoder_decoder_attention_bias"),
                training=training,
                cache=cache)
            logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache
        return symbols_to_logits_fn


class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.
    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
        1. Self-attention layer
        2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, args):
        super(EncoderStack, self).__init__()
        self.args = args
        self.layers = []

        for _ in range(args.enc_layers):
            self_attention_layer = attention_layer.SelfAttention(
                args.hidden_size, args.num_heads, args.dropout)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                args.hidden_size, args.filter_size, args.dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, args),
                PrePostProcessingWrapper(feed_forward_network, args)
            ])

        self.output_normalization = LayerNormalization(args.hidden_size)

    def get_config(self):
        return {
            "args": self.args,
        }

    def call(self, encoder_inputs, attention_bias, inputs_padding, training):
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]
            with tf.name_scope("layer_%d" % n):
                with tf.name_scope("self_attention"):
                    encoder_inputs = self_attention_layer(
                        encoder_inputs, attention_bias, training=training)
                with tf.name_scope("ffn"):
                    encoder_inputs = feed_forward_network(
                        encoder_inputs, training=training)

        return self.output_normalization(encoder_inputs)

class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, args):
        super(DecoderStack, self).__init__()
        self.args = args
        self.layers = []
        for _ in range(args.dec_layers):
            self_attention_layer = attention_layer.SelfAttention(
                args.hidden_size, args.num_heads, args.dropout)
            enc_dec_attention_layer = attention_layer.Attention(
                args.hidden_size, args.num_heads, args.dropout)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                args.hidden_size, args.filter_size, args.dropout)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, args),
                PrePostProcessingWrapper(enc_dec_attention_layer, args),
                PrePostProcessingWrapper(feed_forward_network, args)
            ])
        self.output_normalization = LayerNormalization(args.hidden_size)

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, training, cache=None):
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.name_scope(layer_name):
                with tf.name_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs,
                        decoder_self_attention_bias,
                        training=training,
                        cache=layer_cache)
                with tf.name_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs,
                        encoder_outputs,
                        attention_bias,
                        training=training)
                with tf.name_scope("ffn"):
                    decoder_inputs = feed_forward_network(
                        decoder_inputs, training=training)

        return self.output_normalization(decoder_inputs)

if __name__ == "__main__":
    args = get_args()
    model = Transformer(args, 30)
