### Code from the outliers repository ###
import argparse
import os

import numpy as np
import pandas as pd
import torch
#import wget
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

# from outliers.src.post_processing import cluster_based # whitening, 

import utils as utils

import psutil


### Code from the outliers repository post_processing.py ###
from scipy import cluster as clst
from sklearn.decomposition import PCA


def cluster_based(representations, n_cluster: int, n_pc: int, hidden_size: int=768, seed=42):
    """ Improving Isotropy of input representations using cluster-based method
        adapted from https://github.com/Sara-Rajaee/clusterbased_isotropy_enhancement/

        representations:
            input representations numpy array(n_samples, n_dimension)
        n_cluster:
            the number of clusters
        n_pc:
            the number of directions to be discarded
        hidden_size:
            model hidden size

        returns:
            isotropic representations (n_samples, n_dimension)
    """

    centroids, labels = clst.vq.kmeans2(representations, n_cluster, minit='++', #'points', 
                                    missing='warn', check_finite=True, seed=seed) 
    # Add seed and change initalisation method to kmeans++
    cluster_means = []
    for i in range(max(labels) + 1):
        summ = np.zeros([1, hidden_size])
        for j in np.nonzero(labels == i)[0]:
            summ = np.add(summ, representations[j])
        cluster_means.append(summ / len(labels[labels == i]))

    zero_mean_representations = []
    for i in range(len(representations)):
        zero_mean_representations.append((representations[i]) - cluster_means[labels[i]])

    cluster_representations = {}
    for i in range(n_cluster):
        cluster_representations.update({i: {}})
        for j in range(len(representations)):
            if labels[j] == i:
                cluster_representations[i].update({j: zero_mean_representations[j]})

    # ...why couldn't that have been done in one step?
    cluster_representations2 = []
    for j in range(n_cluster):
        cluster_representations2.append([])
        for key, value in cluster_representations[j].items():
            cluster_representations2[j].append(value)

    # probably unnecessary, gives you dtype object and a deprecation warning
    # cluster_representations2 = np.array(cluster_representations2)

    model = PCA()
    post_rep = np.zeros((representations.shape[0], representations.shape[1]))

    for i in range(n_cluster):
        model.fit(np.array(cluster_representations2[i]).reshape((-1, hidden_size)))
        component = np.reshape(model.components_, (-1, hidden_size))

        for index in cluster_representations[i]:
            sum_vec = np.zeros((1, hidden_size))

            for j in range(min(n_pc, component.shape[0])):
                sum_vec = sum_vec + np.dot(cluster_representations[i][index],
                                           np.expand_dims(np.transpose(component)[:, j], 1)) * component[j]

            post_rep[index] = cluster_representations[i][index] - sum_vec

    return post_rep

### Code from the outliers repository post_processing.py end ###

args = argparse.Namespace(model='glot500', layer=8, device='0', dataset='tatoeba', #tatoeba_use_task_order=True,
                        batch_size=16, save_whitened=False, save_cbie=True, dist='cosine', embed_size=768, tgt_language='en')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, 
                        choices=['xlmr', 'glot500', 'pretrained', 'labse',
                                 'laser', 'distil_xlmr', 'msimcse_glot500'], help='Embedding model')
    parser.add_argument('-s', '--src_lang', type=str, required=True, help='Source language')
    parser.add_argument('-t', '--trg_lang', type=str, required=True, help='Target language')

    return parser.parse_args()


def load_model(args, device):
    model_name = 'cis-lmu/glot500-base' #'xlm-roberta-base'
    model = AutoModel.from_pretrained(model_name) #args.model)
    #model = AutoModelForMaskedLM.from_pretrained(model_name) #args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_name) #args.model)
    model.to(device)
    model.eval()
    print(model.config)
    return model, tokenizer


def mean_pooling(token_embeddings, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeds(data, model, tokenizer, args, device):
    '''Get embeddings for the sentences of the dataset using the chosen model.'''
    tgt_embeddings_layer = None
    attention_masks = None

    dataloader = DataLoader(data, batch_size=args.batch_size, drop_last=False)
    max_len = max([len(x) for x in tokenizer(data, padding=True, truncation=True)['input_ids']])
    print(f'Maximum length: {max_len}')

    i = 0
    tgt_embeddings_list = []
    for batch in tqdm(dataloader):
        encoded = tokenizer(batch, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        encoded.to(device)
        #if i == 0: print(encoded)
        # Calling psutil.cpu_precent() for 4 seconds
        if i % 100 == 0:
            print(f'Test: {i}')
            print('The CPU usage is: ', psutil.cpu_percent(4))
        if attention_masks is None:
            attention_masks = encoded['attention_mask'].cpu()
        else:
            attention_masks = torch.cat((attention_masks, encoded['attention_mask'].cpu()))
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
            #if i == 0: print(output)
            if tgt_embeddings_layer is None:
                #tgt_embeddings_layer = output[2][1:][args.layer].cpu()
                tgt_embeddings_list.append(output[2][1:][args.layer].cpu())
            else:
                #tgt_embeddings_layer = torch.cat((tgt_embeddings_layer, output[2][1:][args.layer].cpu()))
                tgt_embeddings_list.append(output[2][1:][args.layer].cpu())
        #if i == 0: i += 1
        i += 1
        tgt_embeddings_layer = torch.cat(tgt_embeddings_list)
    return mean_pooling(tgt_embeddings_layer, attention_masks).cpu()


def extract_model_embeds(args, dataset_split, device, langs, model, tokenizer):
    '''Extracting embeddings for one language and one model.'''
    model_name = args.model
    print(f'Embeddings from {model_name}.')
    for (src_lang, trg_lang) in langs: # e.g., hsb-de
        print(f"Currently processing {src_lang}-{trg_lang}...")
        try:
            os.makedirs(f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/')
        except OSError:
            print(f"embeddings for {(src_lang, trg_lang)} already exist, skipping")
            #continue

        # DATA TO SPECIFY TRAINING ETC
        data = dataset_split # 'test'
        src = utils.text_to_line(open(f'./UnsupPSE/data/bucc2017/{src_lang}-{trg_lang}/{src_lang}-{trg_lang}.{data}.{src_lang}', 'r').read()) # hsb
        trg = utils.text_to_line(open(f'./UnsupPSE/data/bucc2017/{src_lang}-{trg_lang}/{src_lang}-{trg_lang}.{data}.{trg_lang}', 'r').read()) # de
        #trg = utils.text_to_line(open(f'./data/bucc2017/{src_lang}-{trg_lang}/{src_lang}-{trg_lang}.{data}.{lang}', 'r').read())
        print(len(src), len(trg))
        tgt_embeddings = get_embeds(src, model, tokenizer, args, device)
        print(f'Saving to: ./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{src_lang}_{data}.pt')
        # torch.save(tgt_embeddings, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{trg_lang}_{data}.pt')
        eng_embeddings = get_embeds(trg, model, tokenizer, args, device)
        # # torch.save(tgt_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
        # # torch.save(eng_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng.pt')
        print(f'Saving to: ./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{trg_lang}_{data}.pt')
        # torch.save(eng_embeddings, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{src_lang}_{data}.pt') # de
        # When the embeddings are saved
        # tgt_embeddings = torch.load(f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{trg_lang}_{data}.pt')
        # eng_embeddings = torch.load(f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{src_lang}_{data}.pt')
        
        # For whitening or clustering
        embeddings_list = [tgt_embeddings, eng_embeddings] # hsb, de
        # if args.save_whitened:
        #     print(f'Whitening')
        #     tgt_whitened = torch.Tensor(whitening(tgt_embeddings.numpy()))
        #     eng_whitened = torch.Tensor(whitening(eng_embeddings.numpy()))
        #     torch.save(tgt_whitened, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{trg_lang}_{data}_whitened.pt')
        #     torch.save(eng_whitened, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{src_lang}_{data}_whitened.pt') # de
        if args.save_cbie:
            print(f'CBIE')
            n_cluster = max(len(src) // 300, 1)
            tgt_cbie = torch.Tensor(cluster_based(
                tgt_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=tgt_embeddings.shape[1]))
            eng_cbie = torch.Tensor(cluster_based(
                eng_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=eng_embeddings.shape[1]))
        #     torch.save(tgt_cbie, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{trg_lang}_{data}_cbie.pt')
        #     torch.save(eng_cbie, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{src_lang}_{data}_cbie.pt') # de
            embeddings_list = [tgt_cbie, eng_cbie]
        print(f"Finished saving embeddings for {src_lang}-{trg_lang} in model {model_name}.")
        return embeddings_list


def convert_tf_to_list(tf_embeddings, data_file_path, output_path, start_i=0):
    '''Convert the outlier transformed embeddings into the format for mining.'''
    # tf_embeddings = torch.load(tf_file_path)
    #tf_embeddings_list = tf_embeddings.tolist()
    n = len(tf_embeddings)

    original_data_file = open(data_file_path).read()
    split_data_file = utils.text_to_line(original_data_file)
    assert len(split_data_file) == n, f"File lengths don't match: {len(split_data_file)} and {n}"

    embedding_size = 768
    
    # Initial step
    if start_i == 0:
        sentence = split_data_file[0]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        embedding = tf_embeddings[0].tolist()
        str_embedding = [f'{embed_value:.6f}' for embed_value in embedding]
    
        # First line
        vec_size = embedding_size #len(np_embedding) #len(split_file[0].split(' '))
        
        with open(output_path, 'w', encoding = 'utf8') as out_text:
            out_text.write(f'{n} {vec_size}\n{split_sentence[0]} {" ".join(str_embedding)}\n')

    embedding_list = []
    #for word in sentence_list:
    for i in tqdm(range(start_i + 1, len(split_data_file))):
        sentence = split_data_file[i]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        embedding = tf_embeddings[i].tolist()
        str_embedding = [f'{embed_value:.6f}' for embed_value in embedding]
        if str_embedding: # Not None
            embedding_list.append(f'{split_sentence[0]} {" ".join(str_embedding)}')
        if i % 10000 == 0:
            with open(output_path, 'a', encoding = 'utf8') as out_text:
                out_text.write('\n'.join(embedding_list) + '\n')
            embedding_list = []
    # Remaining lines
    with open(output_path, 'a', encoding = 'utf8') as out_text:
        out_text.write('\n'.join(embedding_list) + '\n')

# Using already extracted sentence embeddings
def read_saved_embeddings(embedding_txt):
    '''Read sentence embeddings that were saved as .txt files.'''
    split_file = utils.text_to_line(embedding_txt)
    print(split_file[0])
    embedding_dim = int(split_file[0].split()[1])
    n = len(split_file)
    sent_id_list = []
    embedding_list = []
    for line in split_file[1:]:
        split_line = line.split(' ')
        sent_id_list.append(split_line[0])
        #assert len(split_line[1:]) == embedding_dim, f'Different embedding dimension: {len(split_line[1:])}, {embedding_dim}'
        embedding_list.append([float(el) for el in split_line[1:]])
    return sent_id_list, embedding_list

def transform_saved_embeddings(args, dataset_split, device, langs, model, tokenizer):
    '''Transform already extracted sentence embeddings'''
    model_name = args.model
    print(f'Embeddings from {model_name}.')
    for (src_lang, trg_lang) in langs: # e.g., hsb-de
        print(f"Currently processing {src_lang}-{trg_lang}...")
        try:
            os.makedirs(f'./embs/{src_lang}-{trg_lang}/{model_name}/approx/')
        except OSError:
            print(f"embeddings for {(src_lang, trg_lang)} already exist, skipping")
            #continue

        # DATA TO SPECIFY TRAINING ETC
        data = dataset_split # 'test'
        embedding_folder_path = f'./UnsupPSE/results_full_xlmr/new_embeddings/doc/bucc2017/{src_lang}-{trg_lang}'
        #e.g.: UnsupPSE/results_full_xlmr/new_embeddings/doc/bucc2017/hsb-de/ glot500.hsb-de.test.de.vec	
        src_file = open(os.path.join(embedding_folder_path, f'{model_name}.{src_lang}-{trg_lang}.{data}.{src_lang}.vec'), 'r').read() # hsb
        trg_file = open(os.path.join(embedding_folder_path, f'{model_name}.{src_lang}-{trg_lang}.{data}.{trg_lang}.vec'), 'r').read() # de
        print(src_file[0:100])

        src_id_list, tgt_embeddings = read_saved_embeddings(src_file) # utils.text_to_line already in the function
        trg_id_list, eng_embeddings = read_saved_embeddings(trg_file)
        id_list = [src_id_list, trg_id_list]
        tgt_embeddings = np.array(tgt_embeddings)
        eng_embeddings = np.array(eng_embeddings)
        
        # For whitening or clustering
        embeddings_list = [tgt_embeddings, eng_embeddings] # hsb, de
        # if args.save_whitened:
        #     print(f'Whitening')
        #     tgt_whitened = torch.Tensor(whitening(tgt_embeddings.numpy()))
        #     eng_whitened = torch.Tensor(whitening(eng_embeddings.numpy()))
        #     torch.save(tgt_whitened, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{trg_lang}_{data}_whitened.pt')
        #     torch.save(eng_whitened, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{src_lang}_{data}_whitened.pt') # de
        if args.save_cbie:
            print(f'CBIE')
            n_cluster = max(len(utils.text_to_line(src_file)) // 300, 1)
            tgt_cbie = torch.Tensor(cluster_based(
                tgt_embeddings, n_cluster=n_cluster, n_pc=12, hidden_size=tgt_embeddings.shape[1]))
            eng_cbie = torch.Tensor(cluster_based(
                eng_embeddings, n_cluster=n_cluster, n_pc=12, hidden_size=eng_embeddings.shape[1]))
        #     torch.save(tgt_cbie, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{trg_lang}_{data}_cbie.pt')
        #     torch.save(eng_cbie, f'./embs/{src_lang}-{trg_lang}/{model_name}/{args.layer}/{src_lang}_{data}_cbie.pt') # de
            embeddings_list = [tgt_cbie, eng_cbie]
        #print(f"Finished saving embeddings for {src_lang}-{trg_lang} in model {model_name}.")
        print('Transformation finished')
        return id_list, embeddings_list

def convert_id_embed_to_list(id_list, embeddings_list, output_path, start_i=0): # data_file_path, 
    '''Convert the outlier transformed embeddings into the format for mining when the ID are separated.'''
    
    n = len(embeddings_list)
    assert len(id_list) == n, f'The lengths of the lists do not match: {n}, {len(id_list)}'
    embedding_size = 768
    
    # Initial step
    if start_i == 0:
        #sentence = split_data_file[0]
        #split_sentence = sentence.split('\t')
        #assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        embedding = embeddings_list[0] #tf_embeddings[0].tolist()
        str_embedding = [f'{embed_value:.6f}' for embed_value in embedding]
    
        # First line
        vec_size = embedding_size #len(np_embedding) #len(split_file[0].split(' '))
        
        with open(output_path, 'w', encoding = 'utf8') as out_text:
            out_text.write(f'{n} {vec_size}\n{id_list[0]} {" ".join(str_embedding)}\n')

    embedding_list = []
    #for word in sentence_list:
    for i in tqdm(range(start_i + 1, n)):
        # sentence = split_data_file[i]
        # split_sentence = sentence.split('\t')
        # assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'
        embedding = embeddings_list[i] #tf_embeddings[i].tolist()

        str_embedding = [f'{embed_value:.6f}' for embed_value in embedding]
        if str_embedding: # Not None
            embedding_list.append(f'{id_list[i]} {" ".join(str_embedding)}')
        if i % 10000 == 0:
            with open(output_path, 'a', encoding = 'utf8') as out_text:
                out_text.write('\n'.join(embedding_list) + '\n')
            embedding_list = []
            
    # Remaining lines
    with open(output_path, 'a', encoding = 'utf8') as out_text:
        out_text.write('\n'.join(embedding_list) + '\n')

def main():
    #args = parse_args()

    model, tokeniser = load_model(args, device='cuda')
    
    # model_name = 'glot500'
    # src_lang = 'hsb'
    # trg_lang = 'de'

    ### Extracting and transforming embeddings ###
    sp_args = parse_args()
    model_name = sp_args.model_name
    src_lang = sp_args.src_lang
    trg_lang = sp_args.trg_lang

    # Using already extracted embeddings
    train_id_list, train_embeddings_list = transform_saved_embeddings(args, dataset_split='train', device='cuda', 
                                        langs=[(src_lang, trg_lang)], model=model, tokenizer=tokeniser)
    test_id_list, test_embeddings_list = transform_saved_embeddings(args, dataset_split='test', device='cuda', 
                                        langs=[(src_lang, trg_lang)], model=model, tokenizer=tokeniser)

    embeddings_dict = {('train', src_lang): (train_id_list[0], train_embeddings_list[0]),
                       ('train', trg_lang): (train_id_list[1], train_embeddings_list[1]),
                       ('test', src_lang): (test_id_list[0], test_embeddings_list[0]),
                       ('test', trg_lang): (test_id_list[1], test_embeddings_list[1])}
    # Converting the embeddings into string vectors
    # for data in ['train', 'test']: # Data split
    #     for lang in [src_lang, trg_lang]: # Language to process
    for (data, lang), (id_list, tf_embeddings) in embeddings_dict.items():
        print(f'Processing the {data} data for {lang}.')
        output_file_name = f'CBIE2_{model_name}.{src_lang}-{trg_lang}.{data}.{lang}.vec'
        convert_id_embed_to_list(id_list=id_list, embeddings_list=tf_embeddings, 
                output_path=f'UnsupPSE/results_full_xlmr/new_embeddings/doc/bucc2017/{src_lang}-{trg_lang}/{output_file_name}', 
                start_i=0)
        #convert_tf_to_list(tf_embeddings, # f'./embs/{trg_lang}-{src_lang}/{model_name}/8/{lang}_{data}_cbie.pt')
    return 0


if __name__ == '__main__':

    main()
