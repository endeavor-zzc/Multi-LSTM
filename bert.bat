set /A a = 1
set /A ALLLINE = 2706
set /A RealAllLine = %ALLLINE%-1

echo blasting...
:loop
if %a% == %ALLLINE% goto end
	rem python extract_features.py -input_file=LLR\left_neg\%a%.seq -output_file=LLR\output_left_neg\%a%.jsonl -vocab_file=vocabs.txt -bert_config_file=bert_large_L-24_H-1024_A-16_I-512\bert_config.json -init_checkpoint=bert_large_L-24_H-1024_A-16_I-512\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8 -max_seq_length=512 -batch_size=64
	rem python jsonl2csv.py LLR\output_left_neg\%a%.jsonl LLR\output_left_neg_csv\%a%.csv
	
	rem python extract_features.py -input_file=LLR\left_pos\%a%.seq -output_file=LLR\output_left_pos\%a%.jsonl -vocab_file=vocabs.txt -bert_config_file=bert_large_L-24_H-1024_A-16_I-512\bert_config.json -init_checkpoint=bert_large_L-24_H-1024_A-16_I-512\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8 -max_seq_length=512 -batch_size=64
	rem python jsonl2csv.py LLR\output_left_pos\%a%.jsonl LLR\output_left_pos_csv\%a%.csv
	
	REM python extract_features.py -input_file=LLR\right_pos\%a%.seq -output_file=LLR\output_right_pos\%a%.jsonl -vocab_file=vocabs.txt -bert_config_file=bert_large_L-24_H-1024_A-16_I-512\bert_config.json -init_checkpoint=bert_large_L-24_H-1024_A-16_I-512\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8 -max_seq_length=512 -batch_size=64
	REM python jsonl2csv.py LLR\output_right_pos\%a%.jsonl LLR\output_right_pos_csv\%a%.csv
	
	REM python extract_features.py -input_file=LLR\right_neg\%a%.seq -output_file=LLR\output_right_neg\%a%.jsonl -vocab_file=vocabs.txt -bert_config_file=bert_large_L-24_H-1024_A-16_I-512\bert_config.json -init_checkpoint=bert_large_L-24_H-1024_A-16_I-512\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8 -max_seq_length=512 -batch_size=64
	REM python jsonl2csv.py LLR\output_right_neg\%a%.jsonl LLR\output_right_neg_csv\%a%.csv
	
	python extract_features.py -input_file=LLR\pos\%a%.seq -output_file=LLR\output_pos\%a%.jsonl -vocab_file=vocabs.txt -bert_config_file=bert_large_L-24_H-1024_A-16_I-512\bert_config.json -init_checkpoint=bert_large_L-24_H-1024_A-16_I-512\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py LLR\output_pos\%a%.jsonl LLR\output_pos_csv\%a%.csv
	
	python extract_features.py -input_file=LLR\neg\%a%.seq -output_file=LLR\output_neg\%a%.jsonl -vocab_file=vocabs.txt -bert_config_file=bert_large_L-24_H-1024_A-16_I-512\bert_config.json -init_checkpoint=bert_large_L-24_H-1024_A-16_I-512\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py LLR\output_neg\%a%.jsonl LLR\output_neg_csv\%a%.csv
	set /A a = %a% + 1
	goto :loop
:end
echo blasting end successfully

pause