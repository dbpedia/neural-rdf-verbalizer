"""The Transformer model from "Attention is all you need"
"""
from __future__ import print_function, absolute_import, division
from six.moves import range

import tensorflow as tf 
#import attention_layer 
#import beam_search 
#import ffn_layer 
#import model_utils

_NEG_INF = -1e9

class Transformer(object):
    """Transformer model for seq2seq data 
    
    Generally consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """

    def __init__(self, params, train):
        """ 
        """