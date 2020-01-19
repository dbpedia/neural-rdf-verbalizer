python preprocess.py --train_src 'data/processed_data/eng/train_src' \
  --train_tgt 'data/processed_data/eng/train_tgt' \
  --eval_src 'data/processed_data/eng/eval_src' \
  --eval_tgt 'data/processed_data/eng/eval_tgt' \
  --test_src 'data/processed_data/eng/test_src' \
  --spl_sym 'data/processed_data/special_symbols' \
  --model transformer --lang eng \
  --vocab_size 16000 --sentencepiece_model 'bpe' --sentencepiece True

python3 'GSoC-19/train_single.py' --train_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/reif_train' \
  --eval_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/reif_eval' \
  --test_path '/content/gdrive/My Drive/data/processed_graphs/eng/gat/reif_test' \
  --src_vocab 'vocabs/gat/eng/reif_src_vocab' \
  --tgt_vocab 'vocabs/gat/eng/train_vocab.model' \
  --batch_size 64 --enc_type gat --dec_type transformer \
  --model gat --vocab_size 16000 \
  --emb_dim 256 --hidden_size 256 \
  --filter_size 512 --use_bias True --beam_size 5 \
  --beam_alpha 0.1 --enc_layers 6 --dec_layers 6 \
  --num_heads 8 --sentencepiece True \
  --steps 150 --eval_steps 500 --checkpoint 1000 \
  --alpha 0.2 --dropout 0.2 \
  --reg_scale 0.0 --learning_rate 0.0001 \
  --lang eng --use_colab True \
  --eval 'GSoC-19/data/processed_data/eng/eval_src' \
  --eval_ref 'GSoC-19/data/processed_data/eng/eval_tgt'
