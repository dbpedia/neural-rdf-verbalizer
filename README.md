﻿# Multilingual RDF verbalizer - GSoC/2019
### Author - [Dwaraknath Gnaneshwar](https://github.com/DwaraknathT)

## Abstract :

This project aims to create a deep-learning-based natural language generation framework that verbalizes RDF triples.

An RDF triple set contains a triple set, each of the form **< subject | predicate | object>**, the model aims to take in a set of such triples and output the information in human-readable form.

A high-level overview of the dataflow would be like this :

![Image](https://raw.githubusercontent.com/DwaraknathT/GSoC-19/final/assets/rdf2nl.png)
[Picture courtesy](https://blog.dbpedia.org/2019/08/08/rdf2nl-generating-texts-from-rdf-data)



For ex :
**< Dwarak | birthplace | Chennai >** **< Dwarak | lives in | India >**
output:
**Dwarak was born in Chennai, and lives in India**
The model must be capable of doing the same in multiple languages, hence the name multilingual RDF verbalizer.

## Model Architecture :
We use attention based encoder-decoder architecture with **Graph Attention Networks** encoder and **Transformer** decoder along with Pure-RNN model and Pure-Transformer model.

The architecture of our model takes the following form.
![Architecture](https://raw.githubusercontent.com/DwaraknathT/GSoC-19/final/assets/architecture.jpg)
[Picture courtesy](https://arxiv.org/pdf/1804.00823.pdf)

The dataset in use is [**WebNLG** challenge's](http://webnlg.loria.fr/pages/challenge.html) dataset.

## Intuition :
We justify the use of Graph Attention Networks by pointing out the fact that in a graph, each node is related to its first order neighbours. While generating the encoded representation, which is passed to the decoder to generate the probability distribution over target vocabulary, we consider each node's features and it's neighbour's features and apply mutual and self attention mechanism over them. The model must be able to culminate these features together and maintain the semantics of triple. By using Graph Networks we inject the sense of structure into the encoders, which is useful when we consider that RDF triples can be maintained and viewed as concepts of Knowledge Graphs.

## Usage :

 - To preprocess the dataset and save graph nodes, edges.

```
python preprocess.py \
  --train_src 'data/processed_data/eng/train_src' \
  --train_tgt 'data/processed_data/eng/train_tgt' \
  --eval_src 'data/processed_data/eng/eval_src' \
  --eval_tgt 'data/processed_data/eng/eval_tgt' \
  --test_src 'data/processed_data/eng/test_src' \
  --spl_sym 'data/processed_data/special_symbols' \
  --model gat --lang eng --sentencepiece True \
  --vocab_size 16000 --sentencepiece_model 'bpe'
```
- To start training with Graph Attention Network encoder and decoder. The preprocessed files are stored in the data folder, use the path in the below code snippet. Please use the hyper-parameters as you see fit, and provide the necessary arguments.
- NOTE: If you use sentencepiece for preprocessing and not specify the flag for training script you may get shape errors. Also, for Transformer, RNN models source and target vocabularies are same.
```
python train_single.py \
  --train_path 'data/processed_graphs/eng/gat/train' \
  --eval_path 'data/processed_graphs/eng/gat/eval' \
  --test_path 'data/processed_graphs/eng/gat/test' \
  --src_vocab 'vocabs/gat/eng/src_vocab' \
  --tgt_vocab 'vocabs/gat/eng/train_vocab.model' \
  --batch_size 1 --enc_type gat --dec_type transformer --model gat --vocab_size 16000 \
  --emb_dim 16 --hidden_size 16 --filter_size 16 --beam_size 5 \
  --beam_alpha 0.1 --enc_layers 1 --dec_layers 1 --num_heads 1 --sentencepiece True \
  --steps 10000 --eval_steps 1000 --checkpoint 1000 --alpha 0.2 --dropout 0.2 \
  --reg_scale 0.0 --decay True --decay_steps 5000 --lang eng --debug_mode False \
  --eval 'data/processed_data/eng/eval_src' --eval_ref 'data/processed_data/eng/eval_tgt'

```
- To train the multilingual model, which concatenates the datasets of individual languages and appends a token for each language's input sentences.
```
python train_multiple.py \
  --train_path 'data/processed_graphs/eng/gat/train' \
  --eval_path 'data/processed_graphs/eng/gat/eval' \
  --test_path 'data/processed_graphs/eng/gat/test' \
  --src_vocab 'vocabs/gat/eng/src_vocab' \
  --tgt_vocab 'vocabs/gat/eng/train_vocab.model' \
  --batch_size 1 --enc_type gat --dec_type transformer \
  --model multi --vocab_size 16000 --emb_dim 16 --hidden_size 16 \
  --filter_size 16 --beam_size 5 --sentencepiece_model 'bpe' --beam_alpha 0.1 \
  --enc_layers 1 --dec_layers 1 --num_heads 1 --sentencepiece True --steps 10000 \
  --eval_steps 1000 --checkpoint 1000 --alpha 0.2 --dropout 0.2 --distillation False \
  --reg_scale 0.0 --decay True --decay_steps 5000 --lang multi --debug_mode False \
  --eval 'data/processed_data/eng/eval_src' --eval_ref 'data/processed_data/eng/eval_tgt'

```

- If you want to train an RNN or Transformer model, Input of the model is .triple and Target is .lex file.

## Use Colab
- To use Google-Colab, set the argument 'use_colab' to True run the following command first then above commands with '!' in front.
```
!git clone https://<github_access_token>@github.com/DwaraknathT/GSoC-19.git
```

- You can get your Github access token from GitHub developer's settings.


- To preprocess the files
```
!python 'GSoC-19/preprocess.py' \
  --train_src 'GSoC-19/data/processed_data/eng/train_src' \
  --train_tgt 'GSoC-19/data/processed_data/eng/train_tgt' \
  --eval_src 'GSoC-19/data/processed_data/eng/eval_src' \
  --eval_tgt 'GSoC-19/data/processed_data/eng/eval_tgt' \
  --test_src 'GSoC-19/data/processed_data/eng/test_src' \
  --spl_sym 'GSoC-19/data/processed_data/special_symbols' \
  --model gat --lang eng --use_colab True \
  --vocab_size 16000 --sentencepiece_model 'bpe' --sentencepiece True

```
- Replace the 'eng' in each parameter with 'ger', 'rus' to process the German and Russian corpus. You can also set sentencepiece to True, and change sentenpiece to 'unigram', 'word'. The vocab size is usually set to 32000, but can be set to anything.

- To start training
```
!python 'GSoC-19/train_single.py' \
  --train_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/train' \
  --eval_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/eval' \
  --test_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/test' \
  --src_vocab 'vocabs/gat/eng/src_vocab' \
  --tgt_vocab 'vocabs/gat/eng/train_vocab.model' \
  --batch_size 64 --enc_type gat --dec_type transformer \
  --model gat --vocab_size 16000 \
  --emb_dim 256 --hidden_size 256 \
  --filter_size 512 --use_bias True --beam_size 5 \
  --beam_alpha 0.1 --enc_layers 6 --dec_layers 6 \
  --num_heads 8 --sentencepiece True \
  --steps 150 --eval_steps 500 --checkpoint 1000 \
  --alpha 0.2 --dropout 0.2 --debug_mode False \
  --reg_scale 0.0 --learning_rate 0.0001 \
  --lang eng --use_colab True \
  --eval 'GSoC-19/data/processed_data/eng/eval_src' \
  --eval_ref 'GSoC-19/data/processed_data/eng/eval_tgt'
```
- If you use sentencepiece the vocab_size argument must match the vocab_size used in the preprocess script. The preprocess file automatically saves the prepcessed datasets as pickle dumps in your drive.

- To run the multilingual model replace train_single.py with train_multiple.py. All languages must be preprocessed to train the multilingual model. The multilingual model preprocesses the data for all languages automatically, no need to change the train_path, eval_path, test_path. the lang, eval and eval_ref parameters must be changed to 'mutli' to save it's checkpoints in a folder of the same name.

## Credits :
- My idea is an extension and a variation of the paper [Deep Graph Convolutional Encoders for
Structured Data to Text Generation](https://arxiv.org/pdf/1810.09995.pdf) , my input pipeline follows the same principle but is a different implementation as comapred to the [paper's.](https://github.com/diegma/graph-2-text)

- My implementation of Transformers is based on the official [Tensorflow's implementation.](https://github.com/tensorflow/models/tree/master/official/transformer) 
