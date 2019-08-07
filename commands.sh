python3 'GSoC-19/train_single.py' --train_path '/content/gdrive/My Drive/data/processed_graphs/rus/gat/reif_train' \
--eval_path '/content/gdrive/My Drive/data/processed_graphs/rus/gat/reif_eval' \
--test_path '/content/gdrive/My Drive/data/processed_graphs/rus/gat/reif_test' \
--src_vocab 'vocabs/gat/rus/reif_src_vocab' --tgt_vocab 'vocabs/gat/rus/train_tgt.model' \
--batch_size 1 --enc_type gat --dec_type transformer --model gat --vocab_size 16000 \
--emb_dim 16 --hidden_size 16  --filter_size 16 --use_bias True --beam_size 5 \
--beam_alpha 0.1  --enc_layers 1 --dec_layers 1 --num_heads 1 --use_edges False \
--steps 1 --eval_steps 1000 --checkpoint 1000 --alpha 0.2 --dropout 0.2 --resume False \
--reg_scale 0.0 --decay True --decay_steps 5000 --lang rus --use_colab True --opt reif \
--eval 'GSoC-19/data/processed_data/rus/eval_src' --eval_ref 'GSoC-19/data/tokenized/rus/eval_tgt'
