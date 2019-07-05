""" Script to train the selected model
    Used to train a Multi-lingual model ( Teacher model )
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import time
import os
from tqdm import tqdm

from data_loader import get_dataset, get_gat_dataset
from src.models import graph_attention_model, rnn_model
from src.utils.model_utils import create_masks, create_transgat_masks
from src.utils.model_utils import loss_function, CustomSchedule
from src.models import transformer
from arguments import get_args
