"""
Inference, take a triple set, load the model and return the sentence
"""
import tensorflow as tf
import networkx as nx
import numpy as np
import pickle
import os

from src.models import graph_attention_model, transformer
from src.utils.model_utils import CustomSchedule, create_transgat_masks
from arguments import get_args


def load_vocabs():
    with open('vocabs/nodes_vocab', 'rb') as f:
        nodes_vocab = pickle.load(f)
    with open('vocabs/target_vocab', 'rb') as f:
        target_vocab = pickle.load(f)

    return nodes_vocab, target_vocab

def load_model(args):
    """
    Function to load the model from stored checkpoint.
    :param args: All arguments that were given to train file
    :type args: Argparse object
    :return: model
    :rtype: tf.keras.Model
    """
    # set up dirs
    if args.use_colab is None:
        output_file = 'results.txt'
        OUTPUT_DIR = 'ckpts'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    else:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/ckpts'
        output_file = OUTPUT_DIR + '/results.txt'
        if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    
    node_vocab, target_vocab = load_vocabs()
    vocab_nodes_size = len(node_vocab.word_index) +1
    vocab_tgt_size = len(target_vocab.word_index) +1

    OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    model = graph_attention_model.TransGAT(args, vocab_nodes_size,
                                           vocab_tgt_size, target_vocab)

    if args.decay is not None:
        learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)
    else:
        optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)
    step = 0

    ckpt = tf.train.Checkpoint(
        model=model
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return model

def process_sentence(line):
    g = nx.MultiDiGraph()
    nodes = []
    triple_list = line.split('< TSP >')
    for l in triple_list:
        l = l.strip().split(' | ')
        g.add_edge(l[0], l[1])
        g.add_edge(l[1], l[0])
        g.add_edge(l[1], l[2])
        g.add_edge(l[2], l[1])
    nodes.append(list(g.nodes))
    array = nx.to_numpy_array(g)
    result = np.zeros((16, 16))
    result[:array.shape[0], :array.shape[1]] = array
    result += np.identity(16)
    nodes_lang, target_lang = load_vocabs()
    node_tensor = nodes_lang.texts_to_sequences(nodes)
    node_tensor = tf.keras.preprocessing.sequence.pad_sequences(node_tensor, padding='post')
    node_paddings = tf.constant([[0, 0], [0, 16-len(nodes[0])]])
    node_tensor = tf.pad(node_tensor, node_paddings, mode='CONSTANT')

    return node_tensor, result

def eval(model, node_tensor, adj):
    """
    Function to carry out the Inference mechanism
    :param model: the model in use
    :type model: tf.keras.Model
    :param node_tensor: input node tensor
    :type node_tensor: tf.tensor
    :param adj: adjacency matrix of node tensor
    :type adj: tf.tensor
    :return: Verbalised sentence
    :rtype: str
    """
    model.trainable = False
    node_vocab, target_vocab = load_vocabs()
    start_token = [target_vocab.word_index['<start>']]
    end_token = [target_vocab.word_index['<end>']]
    dec_input = tf.expand_dims([target_vocab.word_index['<start>']], 0)
    result = ''

    for i in range(82):
        mask = create_transgat_masks(dec_input)
        predictions, attention_weights = model(adj, node_tensor, dec_input, mask)
        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        #predicted_id = tf.argmax(predictions[0]).numpy()
        result += target_vocab.index_word[predicted_id[0][0].numpy() ]+ ' '
        if target_vocab.index_word[predicted_id[0][0].numpy()] == '<end>':
            return result
        #if tf.equal(predicted_id, end_token[0]):
        #    return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        dec_input = tf.concat([dec_input, predicted_id], axis=-1)
        #dec_input = tf.expand_dims([predicted_id], 0)

    return result

def inf(triple, model):
    node_tensor, adj = process_sentence(triple)
    result = eval(model, node_tensor, adj)
    print(result)

if __name__ == "__main__":
    args = get_args()
    f = open(args.eval, 'r')
    model = load_model(args)
    for line in f:
        print(line)
        inf(line, model)
