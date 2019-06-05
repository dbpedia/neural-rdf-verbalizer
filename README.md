# Multilingual RDF verbalizer - GSoC/2019 

##Abstract : 
This project aims to create a deep-learning based natural language generation framework that verbalizes 
RDF triples. 

An RDF triple set contains a triple set, each of the form **< subject | predicate | object >** , the model aims to take in a set of suck triples and output a sentence describing them.


For ex : 
**< Dwarak | birthplace | Chennai >** **< Dwarak | lives in | India >** 
output: 
**Dwarak was born in Chennai, and lives in India**
The model must be capable of doing of in multiple languages. 
##Model : 
We use an attention based encoder-deocder architecture with **Graph Attention Networks** encoder and **Transformer** decoder with changeable modules. 
The dataset in use is **WebNLG** challenge's dataset. 
##Usage : 
 - To preprocess the dataset and save graph nodes, edges and adjacency matrices 
 

        python preprocess.py --path "path_to_triples" --opt adj
 

 - To start training with Graph Attention Network encoder and decoder. The preprocessed files are stored in data folder, use the path in the below code snippet. Please use the hyper-parameters as you see fit, and provide the necessary arguments 
 	
        python train.py --src_path    "path_to_wource.triples" 	\ 
				 --tgt_path    "path_to_target.lex"	 	\
				 --graph_adj   "path_to_adjacency_matrices" 	\
				 --graph_nodes "path_to_graph_nodes"      	\
				 --graph_edges "path_to_graph_edges"	 	\
				 --batch_size --enc_type --dec_type 	 	\
				 --emb_dim --units --hidden_size 	 	\ 
				 --use_bias --num_layers --num_heads 	 	\
				 --use_edges --steps --learning_rate 	 




   
			



  
