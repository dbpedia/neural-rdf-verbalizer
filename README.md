# Multilingual RDF verbalizer - GSoC/2019

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

 - To preprocess the dataset and save graph nodes, edges and adjacency matrices.

```
        python preprocess.py --path "path_to_triples" --opt adj --train True
 ```
 - To start training with Graph Attention Network encoder and decoder. The preprocessed files are stored in the data folder, use the path in the below code snippet. Please use the hyper-parameters as you see fit, and provide the necessary arguments.
```

   	       python train.py 	--src_path    "path_to_source.triples" 	\
				 --tgt_path    "path_to_target.lex"	 	\
				 --graph_adj   "path_to_adjacency_matrices" 	\
				 --graph_nodes "path_to_graph_nodes"      	\
				 --graph_edges "path_to_graph_edges"	 	\
				 --batch_size --enc_type --dec_type 	 	\
				 --emb_dim --enc_units --hidden_size 	 	\
				 --use_bias --num_layers --num_heads 	 	\
				 --use_edges --steps --eval_steps		\
				 --learning_rate --use_colab			\
				 --checkpoint "path_to_checkpoint_dir"

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
	!python 'GSoC-19/preprocess.py' --train_src 'GSoC-19/data/processed_data/eng/train_src' \
	--train_tgt 'GSoC-19/data/processed_data/eng/train_tgt' \
	--eval_src 'GSoC-19/data/processed_data/eng/eval_src' \
	--eval_tgt 'GSoC-19/data/processed_data/eng/eval_tgt' \
	--test_src 'GSoC-19/data/processed_data/eng/test_src' \
	--spl_sym 'GSoC-19/data/processed_data/special_symbols' \
	--model gat --opt reif --lang eng --use_colab True \
	--vocab_size 16000  --max_seq_len 100 --sentencepiece_model 'bpe' --sentencepiece False
```
- Replace the 'eng' in each parameter with 'ger', 'rus' to process the German and Russian corpus. You can also set sentencepiece to True, and change sentenpiece to 'unigram', 'word'. The vocab size is usually set to 32000, but can be set to anything.

- To start training
```
!python3 'GSoC-19/train_single.py' --train_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/reif_train' \
				   --eval_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/reif_eval' \
				   --test_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/reif_test' \
				--src_vocab 'vocabs/gat/eng/reif_src_vocab' \
				--tgt_vocab 'vocabs/gat/eng/train_tgt.model' \
				--batch_size 64 --enc_type gat --dec_type transformer \
				--model gat --vocab_size 16000 \
				--emb_dim 256 --hidden_size 256 \
				--filter_size 512 --use_bias True --beam_size 5 \
				--beam_alpha 0.1  --enc_layers 6 --dec_layers 6 \
				--num_heads 8 --sentencepiece True \
				--steps 150 --eval_steps 500 --checkpoint 1000 \
				--alpha 0.2 --dropout 0.2 \
				--reg_scale 0.0 --learning_rate 0.0001 \
				--lang eng --use_colab True --opt reif \
				--eval 'GSoC-19/data/processed_data/eng/eval_src' \
				--eval_ref 'GSoC-19/data/processed_data/eng/eval_tgt'
```
- If you use sentencepiece the vocab_size argument must match the vocab_size used in the preprocess script. The preprocess file automatically saves the prepcessed datasets as pickle dumps in your drive.

- To run the multilingual model replace train_single.py with train_multiple.py. All languages must be preprocessed to train the multilingual model. The multilingual model preprocesses the data for all languages automatically, no need to change the train_path, eval_path, test_path. the lang, eval and eval_ref parameters must be changed to 'mutli' to save it's checkpoints in a folder of the same name.

## Credits :
- My idea is an extension and a variation of the paper [Deep Graph Convolutional Encoders for
Structured Data to Text Generation](https://arxiv.org/pdf/1810.09995.pdf) , my input pipeline follows the same principle but is a different implementation as comapred to the [paper's authors](https://github.com/diegma/graph-2-text).

- My implementation of Transformers is based on the official [Tensorflow's implementation](https://github.com/tensorflow/models/tree/master/official/transformer) 
