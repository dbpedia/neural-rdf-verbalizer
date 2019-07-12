""" Script to train the selected model
    Used to train a Multi-lingual model ( Teacher model )
    Loads individual student models and then
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

from src.arguments import get_args

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