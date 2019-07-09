"""
Inference, take a triple set, load the model and return the sentence
"""
import tensorflow as tf
import networkx as nx
import numpy as np
import pickle
import os

from src.models import graph_attention_model, transformer
from src.utils.model_utils import CustomSchedule, \
                                create_transgat_masks, create_masks
from arguments import get_args


def load_gat_vocabs():
    with open('vocabs/gat/nodes_vocab', 'rb') as f:
        nodes_vocab = pickle.load(f)
    with open('vocabs/gat/target_vocab', 'rb') as f:
        target_vocab = pickle.load(f)
    with open('vocabs/gat/roles_vocab', 'rb') as f:
        roles_vocab = pickle.load(f)

    return nodes_vocab, roles_vocab, target_vocab

def load_seq_vocabs():
    with open('vocabs/seq2seq/source_vocab', 'rb') as f:
        source_vocab = pickle.load(f)
    with open('vocabs/seq2seq/target_vocab', 'rb') as f:
        target_vocab = pickle.load(f)

    return source_vocab, target_vocab

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
    OUTPUT_DIR += '/' + args.enc_type + '_' + args.dec_type

    if args.enc_type == "gat" and args.dec_type == "transformer":
        node_vocab, roles_vocab, target_vocab = load_gat_vocabs()
        vocab_nodes_size = len(node_vocab.word_index) + 1
        vocab_tgt_size = len(target_vocab.word_index) + 1
        vocab_roles_size = len(roles_vocab.word_index) + 1
        model = graph_attention_model.TransGAT(args, vocab_nodes_size, vocab_roles_size,
                                                vocab_tgt_size, target_vocab)
    elif args.enc_type == 'transformer' and args.dec_type == 'transformer':
        num_layers = args.enc_layers
        num_heads = args.num_heads
        d_model = args.emb_dim
        dff = args.hidden_size
        dropout_rate = args.dropout
        source_vocab, targ_vocab = load_seq_vocabs()
        vocab_inp_size = len(source_vocab.word_index) + 1
        vocab_tgt_size = len(targ_vocab.word_index) + 1
        model = transformer.Transformer(num_layers, d_model, num_heads, dff,
                                        vocab_inp_size, vocab_tgt_size, dropout_rate)
    else:
        node_vocab, roles_vocab, target_vocab = load_gat_vocabs()
        vocab_nodes_size = len(node_vocab.word_index) + 1
        vocab_tgt_size = len(target_vocab.word_index) + 1
        vocab_roles_size = len(roles_vocab.word_index) + 1
        model = graph_attention_model.GATModel(args, vocab_nodes_size,
                                               vocab_roles_size, vocab_tgt_size, target_vocab)
    if args.decay is not None:
        learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)
    else:
        optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                           epsilon=1e-9)

    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    return model

def process_gat_sentence(line):
    g = nx.MultiDiGraph()
    nodes = []
    roles = []
    triple_list = line.split('< TSP >')
    for l in triple_list:
        l = l.strip().split(' | ')
        g.add_edge(l[0], l[1])
        g.add_edge(l[1], l[0])
        g.add_edge(l[1], l[2])
        g.add_edge(l[2], l[1])
    nodes.append(list(g.nodes))
    # set roles
    roles_ = []
    for node in list(g.nodes()):
        role = ''
        for l in triple_list:
            l = l.strip().split(' | ')

            if l[0] == node:
                if role == 'object':
                    role = 'bridge'
                else:
                    role = 'subject'
            elif l[1] == node:
                role = 'predicate'
            elif l[2] == node:
                if role == 'subject':
                    role = 'bridge'
                else:
                    role = 'object'
        roles_.append(role)
    roles.append(roles_)
    array = nx.to_numpy_array(g)
    result = np.zeros((16, 16))
    result[:array.shape[0], :array.shape[1]] = array
    result += np.identity(16)
    nodes_lang, roles_vocab, target_lang = load_gat_vocabs()
    node_tensor = nodes_lang.texts_to_sequences(nodes)
    node_tensor = tf.keras.preprocessing.sequence.pad_sequences(node_tensor, padding='post')
    role_tensor = roles_vocab.texts_to_sequences(roles)
    role_tensor = tf.keras.preprocessing.sequence.pad_sequences(role_tensor, padding='post')

    node_paddings = tf.constant([[0, 0], [0, 16-len(nodes[0])]])
    node_tensor = tf.pad(node_tensor, node_paddings, mode='CONSTANT')
    role_paddings = tf.constant([[0, 0], [0, 16-len(roles[0])]])
    role_tensor = tf.pad(role_tensor, role_paddings, mode='CONSTANT')

    return node_tensor, role_tensor, result

def gat_eval(model, node_tensor, role_tensor, adj):
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
    node_vocab, roles_vocab, target_vocab = load_gat_vocabs()
    start_token = [target_vocab.word_index['<start>']]
    end_token = [target_vocab.word_index['<end>']]
    dec_input = tf.expand_dims([target_vocab.word_index['<start>']], 0)
    result = ''

    for i in range(82):
        mask = create_transgat_masks(dec_input)
        predictions, attention_weights = model(adj, node_tensor, role_tensor, dec_input, mask)
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

def seq2seq_eval(model, triple):
    """
    Function to carry out inference for Transformer model.
    :param model: The model object
    :type model: tf.keras.Model
    :param tensor: preprocessed input tenor of shape [batch_size, seq_length]
    :type tensor: tf.tensor
    :return: The verbalised sentence of the triple
    :rtype: str
    """
    model.trainable = False
    source_vocab, target_vocab = load_seq_vocabs()
    tensor = source_vocab.texts_to_sequences(triple)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    encoder_input = tf.transpose(tensor)
    source_vocab, target_vocab = load_seq_vocabs()
    dec_input = tf.expand_dims([target_vocab.word_index['<start>']], 0)
    result = ''
    for i in range(82):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, dec_input)
        predictions, _ = model(encoder_input, dec_input,
                               True,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask)
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        result += target_vocab.index_word[predicted_id[0][0].numpy()] + ' '
        if target_vocab.index_word[predicted_id[0][0].numpy()] == '<end>':
            return result
        # if tf.equal(predicted_id, end_token[0]):
        #    return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        dec_input = tf.concat([dec_input, predicted_id], axis=-1)
        # dec_input = tf.expand_dims([predicted_id], 0)
    return result

def rnn_eval(args, model, node_tensor, role_tensor, adj):
    model.trainable = False
    node_vocab, roles_vocab, target_vocab = load_gat_vocabs()
    hidden = [tf.zeros((1, args.enc_units))]
    enc_out = model.encoder(node_tensor, adj, role_tensor,
                            args.num_heads, model.trainable, None)
    enc_out_hidden = tf.reshape(enc_out, shape=[enc_out.shape[0], -1])
    enc_hidden = model.hidden(enc_out_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_vocab.word_index['<start>']], 0)
    result = ''
    for t in range(82):
        predictions, dec_hidden, attention_weights = model.decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += target_vocab.index_word[predicted_id] + ' '
        if target_vocab.index_word[predicted_id] == '<end>':
            return result
        dec_input = tf.expand_dims([predicted_id], 0)

    return result 

def inf(args, triple, model):
    if args.enc_type == 'gat' and args.dec_type == 'transformer':
        node_tensor, role_tensor, adj = process_gat_sentence(triple)
        result = gat_eval(model, node_tensor, role_tensor, adj)
        return (result)
    elif args.enc_type == 'transformer' and args.dec_type == 'transformer':
        result = seq2seq_eval(model, triple)
        return result
    else:
        node_tensor, role_tensor, adj = process_gat_sentence(triple)
        result = rnn_eval(args, model, node_tensor, role_tensor, adj)
        return result

if __name__ == "__main__":
    args = get_args()
    model = load_model(args)
    f = open(args.eval, 'r')
    if args.use_colab is True:
        s = open('/content/gdrive/My Drive/data/results.txt', 'w+')
    else:
        s = open('data/results.txt', 'w+')
    #line = 'Point Fortin | country | Trinidad'

    for line in f:
        print(line)
        result = inf(args, line, model)
        print(result)
        s.write(result + '\n')
    #inf (line, model)