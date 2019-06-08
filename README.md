# Multilingual RDF verbalizer - GSoC/2019 

## Abstract : 

This project aims to create a deep-learning based natural language generation framework that verbalizes 
RDF triples. 

An RDF triple set contains a triple set, each of the form **< subject | predicate | object >** , the model aims to take in a set of suck triples and output a sentence describing them.


For ex : 
**< Dwarak | birthplace | Chennai >** **< Dwarak | lives in | India >** 
output: 
**Dwarak was born in Chennai, and lives in India**
The model must be capable of doing of in multiple languages. 

## Model : 
We use an attention based encoder-deocder architecture with **Graph Attention Networks** encoder and **Transformer** decoder with **Pure-RNN** model and **Pure-Transformer** model. 
The dataset in use is [**WebNLG** challenge's](http://webnlg.loria.fr/pages/challenge.html) dataset.

## Usage : 
 - To preprocess the dataset and save graph nodes, edges and adjacency matrices. 
 

        python preprocess.py --path "path_to_triples" --opt adj --train True
 

 - To start training with Graph Attention Network encoder and decoder. The preprocessed files are stored in data folder, use the path in the below code snippet. Please use the hyper-parameters as you see fit, and provide the necessary arguments.
 	
        python train.py --src_path    "path_to_wource.triples" 	\ 
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
				 

- If you want to train an RNN or Transformer model, Input of the model us .triple and Target is .lex file. 
				 
- To use Google-Colab, set the argument 'use_colab' to True run the following command first then above commands with '!' in front. 
	
   
	`!git clone https://<github_access_token>@github.com/DwaraknathT/GSoC-19.git` 
	
- You can get your Github access token in developer's settings. 




  
