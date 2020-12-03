mkdir tokenized
mkdir jsl
mkdir pyt

cd ~/thesis/PreSumm
python3 preprocess.py -mode tokenize -raw_path ~/thesis/raw/ -save_path ~/thesis/tokenized/

python3 preprocess.py -mode custom_format_to_lines -raw_path ~/thesis/tokenized/ -save_path ~/thesis/jsl/ -n_cpus 8 -use_bert_basic_tokenizer false 

python3 preprocess.py -mode custom_format_to_bert -raw_path ~/thesis/jsl/ -save_path ~/thesis/pyt/ -lower -n_cpus 8 -log_file ~/thesis/logs/pytorch_format.log
