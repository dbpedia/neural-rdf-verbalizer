from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import pickle
import sentencepiece as spm
from pathlib import Path
import numpy as np
import os
from src.utils.model_utils import convert, max_length
from src.utils.MultilingualUtils import PreProcess
from src.arguments import get_args

languages = ['eng', 'rus', 'ger']

def ProcessMultlingualDataset(args):
    dataset = {}
    CUR_DIR = os.getcwd()
    levels_up = 0
    DATA_PATH = (os.path.normpath(os.path.join(*([CUR_DIR] + [".."] * levels_up)))) + '/data/processed_data/'

    train_src_dirs = [DATA_PATH + lang + '/train_src' for lang in languages]
    train_tgt_dirs = [DATA_PATH + lang + '/train_tgt' for lang in languages]
    eval_src_dirs = [DATA_PATH + lang + '/eval_src' for lang in languages]
    eval_tgt_dirs = [DATA_PATH + lang + '/eval_tgt' for lang in languages]

    for lang in languages:

        (dataset[lang + '_train_nodes'], dataset[lang + '_train_labels'],
         dataset[lang + '_train_node1'], dataset[lang + '_train_node2']) = PreProcess(DATA_PATH + lang + '/train_src', lang)
        (dataset[lang + '_eval_nodes'], dataset[lang + '_eval_labels'],
         dataset[lang + '_eval_node1'], dataset[lang + '_eval_node2']) = (DATA_PATH + lang + '/eval_src', lang)

    print(len(dataset['eng_train_nodes']))
    exit(0)