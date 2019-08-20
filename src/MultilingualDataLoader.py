from __future__ import absolute_import, division, print_function, unicode_literals

import io
import os

import sentencepiece as spm
import tensorflow as tf

from src.utils.MultilingualUtils import PreProcess
from src.utils.model_utils import PreProcessSentence, _tensorize, Padding as padding

languages = ['eng', 'ger', 'rus']


def LoadMultlingualDataset(args):
    """
    Function to load individual datasets and Preprocess them
     individuall. A language token in also added at
    the start of each dataset.
    Takes in Preprocessed data and trains a sentencepiece model on the
    target sentences if enables, else uses default tensorflow tokenizer.

    :param args: The args obj which contains paths to the preprocessed files
    :type args: ArgParse object
    :return: The mulitlingual dataset, source and target vocab
    :rtype: The multilingual dataset is returned as dict,
            source and tgt vocabs.
    """

    dataset = {}
    CUR_DIR = os.getcwd()
    levels_up = 0
    if args.use_colab is not None:
        DATA_PATH = 'GSoC-19/data/processed_data/'
    else:
        DATA_PATH = (os.path.normpath(os.path.join(*([CUR_DIR] + [".."] * levels_up)))) + '/data/processed_data/'

    # create vocabs for the source
    src_vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
    target_str = ''
    spl_sym = DATA_PATH + 'special_symbols'

    for lang in languages:

        (dataset[lang + '_train_nodes'], dataset[lang + '_train_labels'],
         dataset[lang + '_train_node1'], dataset[lang + '_train_node2']) = PreProcess(DATA_PATH + lang + '/train_src',
                                                                                      lang)
        (dataset[lang + '_eval_nodes'], dataset[lang + '_eval_labels'],
         dataset[lang + '_eval_node1'], dataset[lang + '_eval_node2']) = PreProcess(DATA_PATH + lang + '/eval_src',
                                                                                    lang)
        (dataset[lang + '_test_nodes'], dataset[lang + '_test_labels'],
         dataset[lang + '_test_node1'], dataset[lang + '_test_node2']) = PreProcess(DATA_PATH + lang + '/test_src',
                                                                                    lang)
        train_tgt = io.open(DATA_PATH + lang + '/train_tgt', encoding='UTF-8').read().strip().split('\n')
        dataset[lang + '_train_tgt'] = [(PreProcessSentence(w, args.sentencepiece, lang)) for w in train_tgt]
        eval_tgt = io.open(DATA_PATH + lang + '/eval_tgt', encoding='UTF-8').read().strip().split('\n')
        dataset[lang + '_eval_tgt'] = [(PreProcessSentence(w, args.sentencepiece, lang)) for w in eval_tgt]
        target_str += (DATA_PATH + lang + '/train_tgt') + ','
        target_str += (DATA_PATH + lang + '/eval_tgt') + ','

        # fit the vocab
        src_vocab.fit_on_texts(dataset[lang + '_train_nodes'])
        src_vocab.fit_on_texts(dataset[lang + '_train_labels'])
        src_vocab.fit_on_texts(dataset[lang + '_train_node1'])
        src_vocab.fit_on_texts(dataset[lang + '_train_node2'])
        src_vocab.fit_on_texts(dataset[lang + '_eval_nodes'])
        src_vocab.fit_on_texts(dataset[lang + '_eval_labels'])
        src_vocab.fit_on_texts(dataset[lang + '_eval_node1'])
        src_vocab.fit_on_texts(dataset[lang + '_eval_node2'])

        if args.sentencepiece == 'False':
            src_vocab.fit_on_texts(dataset[lang + '_train_tgt'])
            src_vocab.fit_on_texts(dataset[lang + '_eval_tgt'])

    if args.sentencepiece == 'True':
        os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
        spm.SentencePieceTrainer.Train('--input=' + target_str + ',' + spl_sym + '  \
                                                --model_prefix=vocabs/' + args.model + '/' + args.lang + '/train_tgt \
                                                --vocab_size=' + str(args.vocab_size) + ' --character_coverage=1.0 '
                                                '--model_type=' + args.sentencepiece_model + ' --hard_vocab_limit=false')
        sp = spm.SentencePieceProcessor()
        sp.load('vocabs/' + args.model + '/' + args.lang + '/train_tgt.model')

    if args.sentencepiece == 'True':
        return dataset, src_vocab, sp
    else:
        return dataset, src_vocab, src_vocab


def ProcessMultilingualDataset(args, set=None):
    """
    Takes in the prepocessed Datasets and converts them
    into tensorflow tensors, Adds padding to make the
    targets uniform and packages the individual datasets
    as a combined tf.data.Dataset object.
    Also shuffles and batches the dataset.

    Note : The datasets are not concatenated into one big
    dataset if Knowledge Distillation is being used. We would
    require all datasets seperately to pass each batch through
    both the teacher model and student model. Then the fucntion
    returns a dict with all datasets.

    :param args: Args obj which contains paths to the preprocessed files
    :type args: ArgParse object
    :return: The multilingual dataset along with source and targer vocabs
    and their sizes and maximum target sequence length.
    :rtype: tf.data.Dataset object, vocab objects (src and tgt vocab),
            int ( max sequence length ), int (total buffer size,
            int ( steps per epoch, not much used )
    """

    multilingual_dataset = {}
    dataset, src_vocab, tgt_vocab = LoadMultlingualDataset(args)

    TRAIN_BUFFER_SIZE = 0
    EVAL_BUFFER_SIZE = 0
    for lang in languages:
        if args.sentencepiece == 'False':
            dataset[lang + '_train_tgt'] = _tensorize(src_vocab, dataset[lang + '_train_tgt'])
            dataset[lang + '_eval_tgt'] = _tensorize(src_vocab, dataset[lang + '_eval_tgt'])

        else:
            dataset[lang + '_train_tgt'] = [tgt_vocab.encode_as_ids(w) for w in dataset[lang + '_train_tgt']]
            dataset[lang + '_train_tgt'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_tgt'],
                                                                                         padding='post')
            dataset[lang + '_eval_tgt'] = [tgt_vocab.encode_as_ids(w) for w in dataset[lang + '_eval_tgt']]
            dataset[lang + '_eval_tgt'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_tgt'],
                                                                                        padding='post')

        for part in ['train', 'eval', 'test']:
            dataset[lang + '_' + part + '_nodes'] = padding(
                _tensorize(src_vocab, dataset[lang + '_' + part + '_nodes']), 16)
            dataset[lang + '_' + part + '_labels'] = padding(
                _tensorize(src_vocab, dataset[lang + '_' + part + '_labels']), 16)
            dataset[lang + '_' + part + '_node1'] = padding(
                _tensorize(src_vocab, dataset[lang + '_' + part + '_node1']), 16)
            dataset[lang + '_' + part + '_node2'] = padding(
                _tensorize(src_vocab, dataset[lang + '_' + part + '_node2']), 16)

        TRAIN_BUFFER_SIZE += (dataset[lang + '_train_nodes']).shape[0]
        EVAL_BUFFER_SIZE += (dataset[lang + '_eval_nodes']).shape[0]

    MaxSeqSize = max(dataset['eng_train_tgt'].shape[1],
                     dataset['ger_train_tgt'].shape[1],
                     dataset['rus_train_tgt'].shape[1])

    MULTI_BUFFER_SIZE = 0
    BATCH_SIZE = args.batch_size
    for lang in languages:
        dataset[lang + '_train_tgt'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_tgt'],
                                                          padding='post'), MaxSeqSize)
        dataset[lang + '_eval_tgt'] = padding(
            tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_eval_tgt'],
                                                          padding='post'), MaxSeqSize)

        BUFFER_SIZE = len(dataset[lang + '_train_tgt'])
        MULTI_BUFFER_SIZE += BUFFER_SIZE
        dataset_size = dataset[lang + '_train_tgt'].shape[0]

        for part in ['train', 'eval']:
            if part == 'train':
                multilingual_dataset[lang + '_' + part + '_set'] = tf.data.Dataset.from_tensor_slices(
                    (dataset[lang + '_' + part + '_nodes'],
                     dataset[lang + '_' + part + '_labels'],
                     dataset[lang + '_' + part + '_node1'],
                     dataset[lang + '_' + part + '_node2'],
                     dataset[lang + '_' + part + '_tgt']))
                multilingual_dataset[lang + '_' + part + '_set'] = multilingual_dataset[lang + '_' + part + '_set'].shuffle(BUFFER_SIZE)
            else:
                multilingual_dataset[lang + '_' + part + '_set'] = tf.data.Dataset.from_tensor_slices(
                    (dataset[lang + '_' + part + '_nodes'],
                     dataset[lang + '_' + part + '_labels'],
                     dataset[lang + '_' + part + '_node1'],
                     dataset[lang + '_' + part + '_node2'],
                     dataset[lang + '_' + part + '_tgt']))

        multilingual_dataset[lang + '_test_set'] = tf.data.Dataset.from_tensor_slices(
            (dataset[lang + '_test_nodes'],
             dataset[lang + '_test_labels'],
             dataset[lang + '_test_node1'],
             dataset[lang + '_test_node2']))

    if args.distillation == 'False':
        final_dataset = {}
        for opt in ['train', 'test', 'eval']:
            final_dataset[opt + '_set'] = \
                multilingual_dataset['eng_' + opt + '_set'].concatenate(
                    multilingual_dataset['ger_' + opt + '_set'].concatenate(
                        multilingual_dataset['rus_' + opt + '_set']))

        src_vocab_size = len(src_vocab.word_index) + 1
        tgt_vocab_size = len(src_vocab.word_index) + 1
        final_dataset['train_set'] = final_dataset['train_set'].shuffle(MULTI_BUFFER_SIZE)
        final_dataset['train_set'] = final_dataset['train_set'].batch(BATCH_SIZE,
                                                                      drop_remainder=True)
        # final_dataset['eval_set'] = final_dataset['eval_set'].shuffle(MULTI_BUFFER_SIZE)
        final_dataset['eval_set'] = final_dataset['eval_set'].batch(BATCH_SIZE,
                                                                    drop_remainder=True)
        # final_dataset['test_set'] = final_dataset['train_set'].shuffle(MULTI_BUFFER_SIZE)
        final_dataset['test_set'] = final_dataset['test_set'].batch(BATCH_SIZE,
                                                                    drop_remainder=False)
        steps_per_epoch = int(MULTI_BUFFER_SIZE // BATCH_SIZE)

        print('BUFFER SIZE ' + str(MULTI_BUFFER_SIZE))

        return (final_dataset, src_vocab, src_vocab_size, tgt_vocab,
                tgt_vocab_size, MULTI_BUFFER_SIZE, steps_per_epoch, MaxSeqSize)
    else:
        if args.sentencepiece == 'False':
            src_vocab_size = len(src_vocab.word_index) + 1
            tgt_vocab_size = len(src_vocab.word_index) + 1
        else:
            src_vocab_size = len(src_vocab.word_index) + 1
            tgt_vocab_size = tgt_vocab.get_piece_size()

        steps_per_epoch = int(MULTI_BUFFER_SIZE // BATCH_SIZE)

        for lang in languages:
            multilingual_dataset[lang + '_train_set'] = multilingual_dataset[lang + '_train_set'].batch(BATCH_SIZE,
                                                                                                        drop_remainder=True)
            multilingual_dataset[lang + '_eval_set'] = multilingual_dataset[lang + '_eval_set'].batch(BATCH_SIZE,
                                                                                                      drop_remainder=True)
            multilingual_dataset[lang + '_test_set'] = multilingual_dataset[lang + '_test_set'].batch(BATCH_SIZE,
                                                                                                      drop_remainder=False)

        eval_sets = {}

        for opt in ['test', 'eval']:
            eval_sets[opt + '_set'] = \
                multilingual_dataset['eng_' + opt + '_set'].concatenate(
                    multilingual_dataset['ger_' + opt + '_set'].concatenate(
                        multilingual_dataset['rus_' + opt + '_set']))

        return (multilingual_dataset, src_vocab, src_vocab_size, tgt_vocab,
                tgt_vocab_size, steps_per_epoch, MaxSeqSize)
