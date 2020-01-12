 python preprocess.py --train_src 'data/processed_data/eng/train_src' \
				--train_tgt 'data/processed_data/eng/train_tgt' \
				--eval_src 'data/processed_data/eng/eval_src' \
				--eval_tgt 'data/processed_data/eng/eval_tgt' \
				--test_src 'data/processed_data/eng/test_src' \
				--spl_sym 'data/processed_data/special_symbols' \
				--model transformer --lang eng \
				--vocab_size 16000 --sentencepiece_model 'bpe' --sentencepiece True