python3 preprocess.py --path data/processed_data/train-webnlg-all-notdelex.triple --opt adj --train True

python3 preprocess.py --path data/processed_data/dev-webnlg-all-notdelex.triple --opt adj --train False

python3 train.py --src_path data/processed_data/train-webnlg-all-notdelex.triple --tgt_path data/processed_data/train-webnlg-all-notdelex.lex --graph_adj   data/processed_data/train_graph_pure_adj.npy --graph_nodes data/processed_data/train_graph_nodes --graph_edges data/processed_data/train_graph_edges --graph_roles data/processed_data/train_node_roles --use_edges True --enc_type gat --dec_type transformer --batch_size 80 --emb_dim 512 --hidden_size 512 --enc_layers 1 --dec_layers 1 --num_heads 8 --epochs 10 --eval_steps 160 --checkpoint 160 --dropout 0.2