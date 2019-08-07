# Multilingual RDF verbalizer - GSoC/2019 

## Abstract : 

This project aims to create a deep-learning based natural language generation framework that verbalizes 
RDF triples. 

An RDF triple set contains a triple set, each of the form **< subject | predicate | object >** , the model aims to take in a set of such triples and output a sentence describing them.


For ex : 
**< Dwarak | birthplace | Chennai >** **< Dwarak | lives in | India >** 
output: 
**Dwarak was born in Chennai, and lives in India**
The model must be capable of doing of in multiple languages. We add an artificial language token at the beginning of each triple 
to distinguish between models while training.  

## Preprocessing :
We have two kinds of preprocessing for the triples based on the type of models you are willing to use. 
- To preprocess the dataset and save graph nodes, edges and edge labels, run the following command. Change 'rus' to 'ger', 'eng' to start peprocessing German and English datasets. 
```
 python 'GSoC-19/preprocess.py' --train_src 'GSoC-19/data/processed_data/rus/train_src' \
				--train_tgt 'GSoC-19/data/processed_data/rus/train_tgt' \
				--eval_src 'GSoC-19/data/processed_data/rus/eval_src' \
				--eval_tgt 'GSoC-19/data/processed_data/rus/eval_tgt' \
				--test_src 'GSoC-19/data/processed_data/rus/test_src' \
				--model gat --opt reif --lang rus --use_colab True 
```
## To use [SentencePiece tokenizer](https://github.com/google/sentencepiece) and vocab trainer : 

```
 python 'GSoC-19/preprocess.py' --train_src 'GSoC-19/data/processed_data/rus/train_src' \
				--train_tgt 'GSoC-19/data/processed_data/rus/train_tgt' \
				--eval_src 'GSoC-19/data/processed_data/rus/eval_src' \
				--eval_tgt 'GSoC-19/data/processed_data/rus/eval_tgt' \
				--test_src 'GSoC-19/data/processed_data/rus/test_src' \
				--model gat --opt reif --lang rus --use_colab True \
				--vocab_size 16000  --max_seq_len 100 --sentencepiece_model 'bpe'
```
## Model : 
We use an attention based encoder-deocder architecture with **Graph Attention Networks** encoder and **Transformer** decoder with **Pure-RNN** model and **Pure-Transformer** model. 
The dataset in use is [**WebNLG** challenge's](http://webnlg.loria.fr/pages/challenge.html) dataset.

## Usage : 

 - To start training with Graph Attention Network encoder and decoder. The preprocessed files are stored in data folder, use the path in the below code snippet. Please use the hyper-parameters as you see fit, and provide the necessary arguments.
 
 ```
  python3 train_single.py  --train_path 'Path to train file in processed_graphs' \
				--eval_path 'Path to dev file in processed_graphs' \
				--test_path 'Path to test file in processed_graphs' \
				--src_vocab 'vocabs/gat/rus/reif_src_vocab' 
				--tgt_vocab 'vocabs/gat/rus/train_tgt.model' \
				--batch_size 32 --enc_type gat --dec_type transformer \
				--model gat --vocab_size 16000 --emb_dim 512 \
				--hidden_size 512  --filter_size 768 --use_bias True --beam_size 5 \
				--beam_alpha 0.1  --enc_layers 6 --dec_layers 6 \
				--num_heads 8 --use_edges False --steps 50000 \
				--eval_steps 1000 --checkpoint 1000 --alpha 0.2 \
				--dropout 0.2 --resume False --reg_scale 0.0 \
				--decay True --decay_steps 5000 \
				--lang rus --use_colab True --opt reif \
				--eval 'Path to eval source file' \
				--eval_ref 'Path to eval targets'
```
				 

- Or, you could put the command in a bash file and run 
```
bash commands.sh
```
			
## Use Colab 
- To use Google-Colab, set the argument 'use_colab' to True run the following command first then above commands with '!' in front. 
	
   
	`!git clone https://<github_access_token>@github.com/DwaraknathT/GSoC-19.git` 
	
- You can get your Github access token in developer's settings. 


