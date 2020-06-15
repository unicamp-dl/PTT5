import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help='Input file')
parser.add_argument('-m', '--model_prefix', type=str, required=True, help='Prefix for model and vocab files')

args = parser.parse_args()

fixed_args_list = [
                 f'--input={args.input}',
                f'--model_prefix={args.model_prefix}',
                '--vocab_size=32000',
                '--input_sentence_size=2000000',
                '--shuffle_input_sentence=true',
# '--pad_id=0 --unk_id=1 --eos_id=2 --pad_piece=<pad> --unk_piece=<unk> --eos_piece=</s> bos_id=-1',
                '--pad_id=0',
                '--eos_id=1',
                '--unk_id=2',
                '--bos_id=-1',
#                '--character_coverage=0.99995',
                '--character_coverage=1.0',
                '--model_type=unigram',
#                '--user_defined_symbols=R$'

    ]

args_join = ' '.join(fixed_args_list)
print(args_join)
spm.SentencePieceTrainer.train(args_join)
