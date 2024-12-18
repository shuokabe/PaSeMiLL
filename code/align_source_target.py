import argparse
import re
import sentencepiece
import simalign

from tqdm import tqdm


# Choose layer
LAYER = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--margin_file', type=str, required=True, help='Margin score file')
    parser.add_argument('-s', '--source_file', type=str, required=True, help='Source language file')
    parser.add_argument('-t', '--target_file', type=str, required=True, help='Target language file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file path')

    return parser.parse_args()


def simple_preprocess(sentence):
    '''Simple preprocessing for sentences.'''
    new_sentence = re.sub(r'[.,";?!]', '', sentence)
    return new_sentence

def text_to_dict(text):
    '''Convert the corpus into a dictionary'''
    split_text = text.split('\n')[:-1]
    text_dict = dict()
    for line in tqdm(split_text):
        split_line = line.split('\t')
        assert len(split_line) == 2, f'More than two elements: {len(split_line)}'
        #text_dict[split_line[0]] = simple_preprocess(split_line[1])
        text_dict[split_line[0]] = split_line[1]
    return text_dict


# Aligning sentences
def align_source_target(sentence_pair_list, source_dict, target_dict):
    '''Align filtered source and target corpora.'''
    split_sentence_pair_list = [line.split('\t') for line in sentence_pair_list]
    # SimAlign alignment model
    align_model = simalign.SentenceAligner(model='xlm-roberta-base', token_type='word', matching_methods='a', distortion=0.0, device='cuda', layer=LAYER) 
    alignement_list = []
    split_sentence_pair_list = split_sentence_pair_list#[0:10]
    for sentence_pair in tqdm(split_sentence_pair_list):
        source_sent, target_sent = source_dict[sentence_pair[0]], target_dict[sentence_pair[1]]
        #print(source_sent, target_sent)
        align_sent = align_model.get_word_aligns(source_sent, target_sent)['inter']
        #align_sent_bwd = align_model.get_word_aligns(target_sent, source_sent)['inter'] symmetrical so not needed
        #print(align_sent)
        alignement_list.append(align_sent) #[align_sent_fwd, align_sent_bwd])
    return alignement_list

def align_percentage(alignment_list):
    '''Compute the alignment coverage for each sentence pair.'''
    align_rate_list = []
    n = len(alignment_list)
    for i in tqdm(range(n)):
        #sentence_pair = sentence_pair_list[i]
        alignment = alignment_list[i]
        align_rate_fwd = len(alignment) / (alignment[-1][0] + 1) # Largest index in sentence
        max_target_id = max([alignment_pair[1] for alignment_pair in alignment])
        align_rate_bwd = len(alignment) / (max_target_id + 1)
        align_rate = 0.5 * (align_rate_fwd + align_rate_bwd)
        align_rate_list.append(align_rate)
    return align_rate_list

def token_list_length(token_list):
    '''Compute the character length of a token list.'''
    reconst_token_list = [''.join(sub_token_list) for sub_token_list in token_list]
    return sum([len(token) for token in reconst_token_list])

def align_len_sentence_percentage(alignment_list, sentence_pair, source_dict, target_dict, tokeniser):
    '''Compute the alignement coverage for a given sentence pair and alignment.'''
    split_sentence_pair = sentence_pair.split('\t')
    src_sent, trg_sent = source_dict[split_sentence_pair[0]], target_dict[split_sentence_pair[1]]
    #split_src_sent, split_trg_sent = src_sent.split(' '), trg_sent.split(' ')
    split_src_sent = [tokeniser.tokenize(word) for word in src_sent.split()]
    split_trg_sent = [tokeniser.tokenize(word) for word in trg_sent.split()]
    align_src_list, align_trg_list = [], []
    print(split_src_sent, split_trg_sent, alignment_list)
    for (src_align_idx, trg_align_idx) in alignment_list:
        align_src, align_trg = split_src_sent[src_align_idx], split_trg_sent[trg_align_idx]
        align_src_list.append(align_src)
        align_trg_list.append(align_trg)
    # Compute character length ratio
    align_src_length = token_list_length(align_src_list)
    align_trg_length = token_list_length(align_trg_list)
    src_length = token_list_length(split_src_sent)
    trg_length = token_list_length(split_trg_sent)
    # For both directions
    align_rate_fwd = align_src_length / src_length
    align_rate_bwd = align_trg_length / trg_length
    return 0.5 * (align_rate_fwd + align_rate_bwd)

def align_length_percentage(alignment_list, sentence_pair_list, source_dict, target_dict):
    '''Compute the alignment coverage for each sentence pair, weighted by the token length.'''
    align_rate_list = []
    n = len(alignment_list)
    tokeniser = simalign.EmbeddingLoader(model='xlm-roberta-base', device='cuda', layer=LAYER).tokenizer
    for i in tqdm(range(n)):
        sentence_pair = sentence_pair_list[i]
        alignment = alignment_list[i]
        align_rate = align_len_sentence_percentage(alignment, sentence_pair, source_dict, target_dict, tokeniser)
        align_rate_list.append(align_rate)
    return align_rate_list

def align_rate_file(sentence_pair_list, alignment_list, src_train_dict, trg_train_dict):
    '''Create a new file with the alignment rate for each sentence pair.'''
    #align_rate_list = align_percentage(alignment_list)
    align_rate_list = align_length_percentage(alignment_list, sentence_pair_list, src_train_dict, trg_train_dict)
    n = len(align_rate_list)
    assert n == len(sentence_pair_list), f'The two lists do not have the same size: {n, len(sentence_pair_list)}.'
    new_file_list = []
    for i in tqdm(range(n)):
        sentence_pair = sentence_pair_list[i]
        new_line_list = [sentence_pair, f'{align_rate_list[i]:.6f}']
        new_file_list.append('\t'.join(new_line_list))
    return new_file_list


def main():
    args = parse_args()

    # Input files
    src_file = open(args.source_file, 'r').read()
    trg_file = open(args.target_file, 'r').read()
    src_file_dict = text_to_dict(src_file)
    trg_file_dict = text_to_dict(trg_file)

    margin_train_file = open(args.margin_file, 'r').read() 
    split_margin_train = margin_train_file.split('\n')[:-1]

    align_list_train = align_source_target(split_margin_train, src_file_dict, trg_file_dict) 
    align_rate_train_file = align_rate_file(split_margin_train, align_list_train, src_file_dict, trg_file_dict) 

    with open(args.output_file, 'w') as file:
        file.write('\n'.join(align_rate_train_file))


if __name__ == '__main__':
    main()
