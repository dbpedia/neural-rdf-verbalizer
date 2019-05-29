""" Encoder-base class """

from __future__ import absolute_import 
from __future__ import division, print_function

import abc 
from collections import namedtuple
import six 

from mvb.configurable import Configurable 
from mvb.graph_module import GraphModule 

EncoderOutput = namedtuple()