python3 preprocess.py --train_src 'data/processed_data/eng/train_src' \
--train_tgt 'data/processed_data/eng/train_tgt' \
--eval_src 'data/processed_data/eng/eval_src' \
--eval_tgt 'data/processed_data/eng/eval_tgt' \
--test_src 'data/processed_data/eng/test_src' \
--spl_sym 'data/processed_data/metadata/special_symbols' \
--model gat --lang eng --sentencepiece True \
--vocab_size 16000 --max_seq_len 100 --sentencepiece_model 'bpe'
