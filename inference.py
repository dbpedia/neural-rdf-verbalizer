"""
Inference, take a triple set, load the model and return the sentence
"""
import tensorflow as tf
import networkx as nx
import numpy as np
import pickle
import os
from nltk.translate.bleu_score import corpus_bleu

from src.models import graph_attention_model, transformer
from src.utils.model_utils import CustomSchedule, create_transgat_masks
from src.arguments import get_args
from src.utils.rogue import rouge_n

def load_gat_vocabs(lang):
    with open('vocabs/gat/'+lang+'/src_vocab', 'rb') as f:
        src_vocab = pickle.load(f)
    with open('vocabs/gat/'+lang+'/target_vocab', 'rb') as f:
        target_vocab = pickle.load(f)

    return src_vocab, target_vocab

def load_seq_vocabs():
    with open('vocabs/seq2seq/vocab', 'rb') as f:
        vocab = pickle.load(f)

    return vocab

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
        node_vocab, target_vocab = load_gat_vocabs(args.lang)
        vocab_nodes_size = len(node_vocab.word_index) + 1
        vocab_tgt_size = len(target_vocab.word_index) + 1
        model = graph_attention_model.TransGAT(args, vocab_nodes_size, vocab_tgt_size,target_vocab)

    elif args.enc_type == 'transformer' and args.dec_type == 'transformer':
        num_layers = args.enc_layers
        num_heads = args.num_heads
        d_model = args.emb_dim
        dff = args.hidden_size
        dropout_rate = args.dropout
        vocab = load_seq_vocabs()
        vocab_size = len(vocab.word_index) + 1
        model = transformer.Transformer(args, vocab_size)

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

def process_gat_sentence(line, src_lang, target_lang, lang):
    g = nx.MultiDiGraph()
    nodes = []
    labels = []
    node1 = []
    node2 = []
    temp_node1 = []
    temp_node2 = []
    temp_label = []

    triple_list = line.split('< TSP >')
    for l in triple_list:
        l = l.strip().split(' | ')
        #l = ['<'+lang+'> ' + x for x in l]
        g.add_edge(l[0], l[1], label='A_ZERO')
        g.add_edge(l[1], l[2], label='A_ONE')
    node_list = list(g.nodes())
    node_list.append(lang)
    print(node_list)
    nodes.append(node_list)
    edge_list = list(g.edges.data())
    for edge in edge_list:
        temp_node1.append(edge[0])
        temp_node2.append(edge[1])
        label = (edge[2]['label'])
        temp_label.append(label)
    node1.append(temp_node1)
    node2.append(temp_node2)
    labels.append(temp_label)
    # set roles
    node_tensor = src_lang.texts_to_sequences(nodes)
    node_tensor = tf.keras.preprocessing.sequence.pad_sequences(node_tensor, padding='post')
    label_tensor = src_lang.texts_to_sequences(labels)
    label_tensor = tf.keras.preprocessing.sequence.pad_sequences(label_tensor, padding='post')
    node1_tensor = src_lang.texts_to_sequences(node1)
    node1_tensor = tf.keras.preprocessing.sequence.pad_sequences(node1_tensor, padding='post')
    node2_tensor = src_lang.texts_to_sequences(node2)
    node2_tensor = tf.keras.preprocessing.sequence.pad_sequences(node2_tensor, padding='post')

    node_paddings = tf.constant([[0, 0], [0, 16 - node_tensor.shape[1]]])
    node_tensor = tf.pad(node_tensor, node_paddings, mode='CONSTANT')
    label_padding = tf.constant([[0, 0], [0, 16 - label_tensor.shape[1]]])
    label_tensor = tf.pad(label_tensor, label_padding, mode='CONSTANT')
    node1_padding = tf.constant([[0, 0], [0, 16 - node1_tensor.shape[1]]])
    node1_tensor = tf.pad(node1_tensor, node1_padding, mode='CONSTANT')
    node2_padding = tf.constant([[0, 0], [0, 16 - node2_tensor.shape[1]]])
    node2_tensor = tf.pad(node2_tensor, node2_padding, mode='CONSTANT')

    return node_tensor, label_tensor, node1_tensor, node2_tensor

def gat_eval(model, node_tensor, label_tensor,
             node1_tensor, node2_tensor, src_vocab, target_vocab):
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
    start_token = [target_vocab.word_index['<start>']]
    end_token = [target_vocab.word_index['<end>']]
    dec_input = tf.expand_dims([target_vocab.word_index['<start>']], 0)
    result = ''
    '''
    for i in range(82):
        mask = create_transgat_masks(dec_input)
        predictions = model(node_tensor, label_tensor, node1_tensor, node2_tensor, targ=dec_input, mask=None)
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
    '''
    predictions = model(node_tensor, label_tensor, node1_tensor, node2_tensor, targ=None, mask=None)
    pred = (predictions['outputs'][0].numpy())
    for i in pred:
        if i == 0:
            continue
        if ((target_vocab.index_word[i] != '<start>')):
            result += target_vocab.index_word[i] + ' '
        if (target_vocab.index_word[i] == '<end>'):
            return result
    #'''
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
    source_vocab = load_seq_vocabs()
    tensor = source_vocab.texts_to_sequences(triple)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    encoder_input = tf.transpose(tensor)
    vocab = load_seq_vocabs()
    dec_input = tf.expand_dims([vocab.word_index['<start>']], 0)
    result = ''
    '''
    for i in range(82):
        predictions= model(inputs=encoder_input, targets=None, training=False)
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        result += vocab.index_word[predicted_id[0][0].numpy()] + ' '
        if vocab.index_word[predicted_id[0][0].numpy()] == '<end>':
            return result
        # if tf.equal(predicted_id, end_token[0]):
        #    return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        dec_input = tf.concat([dec_input, predicted_id], axis=-1)
        # dec_input = tf.expand_dims([predicted_id], 0)
    '''
    predictions = model(encoder_input, targets=None, training=model.trainable)
    pred = (predictions['outputs'][0].numpy())
    for i in pred:
        if i==0:
            continue
        if ((vocab.index_word[i] != '<start>') or (vocab.index_word[i] != '<end>')):
            result += vocab.index_word[i] + ' '
        if (vocab.index_word[i] == '<end>'):
            return result

    return result

def rnn_eval(args, model, node_tensor, role_tensor, adj):
    model.trainable = False
    node_vocab, roles_vocab, target_vocab = load_gat_vocabs()
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

def inf(args, triple, model, src_vocab, target_vocab):
    if args.enc_type == 'gat' and args.dec_type == 'transformer':
        node_tensor, label_tensor, node1_tensor, node2_tensor = process_gat_sentence(triple, src_vocab, target_vocab, args.lang)
        result = gat_eval(model, node_tensor, label_tensor, node1_tensor, node2_tensor, src_vocab, target_vocab)
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
    verbalised_triples = []
    if args.enc_type == 'gat':
        src_vocab, target_vocab = load_gat_vocabs(args.lang)
    for i,line in enumerate(f):
        if i< args.num_eval_lines:
            print(line)
            result = inf(args, line, model, src_vocab, target_vocab)
            result = result.strip('<start>')
            result = result.strip('<end>')
            verbalised_triples.append(result)
            print(result)
            s.write(result + '\n')
    #inf (line, model)
    ref_sentence = []
    reference = open(args.eval_ref, 'r')
    for i, line in enumerate(reference):
        if ( i< len(verbalised_triples)):
            ref_sentence.append(line)
    print('Rogue '+ str(rouge_n(verbalised_triples, ref_sentence))+'\n')
    score = corpus_bleu(ref_sentence, verbalised_triples)
    print('BLEU ' + str(score)+'\n')