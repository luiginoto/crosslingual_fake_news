import os
import json
import argparse
import shutil
from utils import *
from collections import Counter

def preprocess_covid(data_dir, save_dir, train_langs=['en', 'es', 'pt', 'it'], val_langs=['fr'], test_langs=['hi']):
    langs_split_dict = {}
    
    for lang in train_langs:
        langs_split_dict[lang] = 'train'
    for lang in val_langs:
        langs_split_dict[lang] = 'val'
    for lang in test_langs:
        langs_split_dict[lang] = 'test'
    
    langs = set(train_langs + val_langs + test_langs)
    langs_cnt = Counter()
    
    for line in open(data_dir + '/news_collection.json', 'r'):
        item = json.loads(line)
        try:
            label = item['label']
            lang = item['lang']
            text = clean_sentence(item['ref_source']['text'])
            if (text != '') and (lang in langs) and (label in ['real', 'fake']):
                save_sample_covid(item, save_dir, lang, langs_split_dict, f'sample_{langs_cnt[lang]}')
                langs_cnt[lang] += 1
        except:
            continue
    
def ranker_parse_args():
    parser = argparse.ArgumentParser(description='MM-COVID data preprocessing')
    parser.add_argument('--data_dir', type=str, default='../../Datasets/mm_covid', help='location of the unprocessed MM-COVID dataset')
    parser.add_argument('--save_dir', type=str, default='../datasets/Dataset', help='directory where processed Amazon dataset is saved')
    parser.add_argument('--train_langs', type=eval, default='["en", "es", "pt", "it"]', help='list of meta-train languages of MM-COVID dataset')
    parser.add_argument('--val_langs', type=eval, default='["fr"]', help='list of meta-val languages of MM-COVID dataset')
    parser.add_argument('--test_langs', type=eval, default='["hi"]', help='list of meta-test languages of MM-COVID dataset')
    return parser.parse_args()

if __name__ == '__main__':

    args = ranker_parse_args()

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

    preprocess_covid(args.data_dir, args.save_dir, args.train_langs, args.val_langs, args.test_langs)

