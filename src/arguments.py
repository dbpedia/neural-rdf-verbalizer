""" File to hold arguments """
import argparse
# data arguments

parser = argparse.ArgumentParser(description="Main Arguments")

# model paramteres
parser.add_argument(
    '--enc_type', default='rnn', type=str, required=True,
    help='Type of encoder Transformer | gat | rnn')
parser.add_argument(
    '--dec_type', default='rnn', type=str, required=True,
    help='Type of decoder Transformer | rnn')
parser.add_argument(
    '--model', default='gat', type=str, required=True,
    help='Model type')
parser.add_argument(
    '--opt', type=str, required=False, help='The mode in which GAT model is operated -> \
                                             Use Roles method or Reification (roles, reif)')
parser.add_argument(
    '--train', type=bool, required=False, help='In training mode or eval mode')
parser.add_argument(
    '--distillation', type=bool, required=False, help='To use Knowledge Distilaltion in the'
                                                      'multilingual model')
parser.add_argument(
    '--resume', type=bool, required=True, help='Yes | no, to resume training')

# Colab options
parser.add_argument(
    '--use_colab', type=bool, required=False, help='Use Google Colab or not')

# preprocess arguments
parser.add_argument(
    '--train_path', type=str, required=True, help='Path to train dataset')
parser.add_argument(
    '--eval_path', type=str, required=True, help='Path to eval dataset')
parser.add_argument(
    '--test_path', type=str, required=True, help='Path to test dataset')
parser.add_argument(
    '--src_vocab', type=str, required=True, help='Path to Vocab of the dataset')
parser.add_argument(
    '--tgt_vocab', type=str, required=True, help='Path to Vocab of the dataset')
parser.add_argument(
    '--lang', type=str, required=True, help='Lang of source and target files')
parser.add_argument(
    '--eval', type=str, required=False, help='Path to Lex file of the Eval set')
parser.add_argument(
    '--eval_ref', type=str, required=False, help='Path to Lex file of the Eval set')
parser.add_argument(
    '--num_eval_lines', type=int, required=False, help='Number of sentences to be used to eval')

# training parameters
parser.add_argument(
    '--steps', type=int, required=False, help='Number of training steps')
parser.add_argument(
    '--eval_steps', type=int, required=False, help='Evaluate every x steps')
parser.add_argument(
    '--checkpoint', type=int, required=False, help='Save checkpoint every these steps')
parser.add_argument(
    '--checkpoint_dir', type=str, required=False, help='Path to checkpoints')

parser.add_argument(
    '--epochs', type=int, default=None,
        required=False, help='Number of epochs (deprecated)')
parser.add_argument(
    '--batch_size', type=int, required=True, help='Batch size')
parser.add_argument(
    '--vocab_size', type=int, required=True, help='Vocab Size for the multilingual model')
parser.add_argument(
    '--emb_dim', type=int, required=True, help='Embedding dimension')
parser.add_argument(
    '--hidden_size', type=int, required=True, help='Size of hidden layer output')
parser.add_argument(
    '--filter_size', type=int, required=True, help='Size of FFN Filters ')
parser.add_argument(
    '--enc_layers', type=int, required=True, help='Number of layers in encoder')
parser.add_argument(
    '--dec_layers', type=int, required=True, help='Number of layers in decoder')
parser.add_argument(
    '--num_heads', type=int, required=True, help='Number of heads in self-attention')
parser.add_argument(
    '--use_bias', type=bool, required=False, help='Add bias or not')
parser.add_argument(
    '--use_edges', type=bool, required=False, help='Add edges to embeddings')
parser.add_argument(
    '--dropout', type=float, required=False, help='Dropout rate')
parser.add_argument(
    '--reg_scale', type=float, required=False, help='L2 Regularizer scale')
parser.add_argument(
    '--enc_units', type=int, required=False, help='Number of encoder units')
parser.add_argument(
    '--num_examples', default=None, type=int, required=False,
    help='Number of examples to be processed')
parser.add_argument(
    '--tensorboard', type=bool, required=False, help='Use tensorboard or not')
parser.add_argument(
    '--colab', type=bool, required=False, help='Use Google-Colab')

# hyper-parameters
parser.add_argument(
    '--optimizer', type=str, required=False, help='Optimizer that will be used')
parser.add_argument(
    '--alpha', type=float, required=False, default= 0.2, help='Alpha value for LeakyRELU')
parser.add_argument(
    '--beam_size', type=int, required=False, default= 0.2, help='Beam search size ')
parser.add_argument(
    '--beam_alpha', type=float, required=False, default= 0.2, help='Alpha value for Beam search')
parser.add_argument(
    '--loss', type=str, required=False, help='Loss function to calculate loss')
parser.add_argument(
    '--learning_rate', type=float, required=False, help='Learning rate')
parser.add_argument(
    '--decay', type=bool, required=False, help='Use learning rate decay')
parser.add_argument(
    '--decay_rate', type=float, required=False, help='Decay rate ')
parser.add_argument(
    '--decay_steps', type=int, required=False, help='Decay every this steps ')
parser.add_argument(
    '--scheduler_step', type=int, required=False, help='Step to start learning rate scheduler')

#inference parameters

def get_args():
    args = parser.parse_args()
    return args