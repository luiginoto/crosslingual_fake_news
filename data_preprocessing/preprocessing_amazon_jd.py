import os
import json
from xml.etree.cElementTree import iterparse
import codecs
import re
import random
import argparse
import shutil

def save_sample_amazon(item, save_dir, lang, langs_split_dict, filename):
    outdict = {}
    class_name = "neg" if float(item.find("rating").text) < 3 else "pos"
    outdict["source_sentence"] = "dummy"
    outdict["target_sentence"] = item.find("text").text
    outdict["source"] = 'Amazon'
    outdict["teacher_encoding"] = [1, 0] if float(item.find("rating").text) < 3 else [0, 1]
    outdict["teacher_name"] = "ground_truth"
    outdict["target_language"] = lang
    file_save_dir = f"{save_dir}/{langs_split_dict[lang]}/Amazon/{lang}/{class_name}"
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir)
    with open(f"{file_save_dir}/{filename}.json", "w", encoding='utf-8') as outfile:
        json.dump(outdict, outfile)# , ensure_ascii=True)


def save_sample_jd(item, save_dir, data_role, filename):
    outdict = {}
    class_name = "neg" if float(item['score']) < 3 else "pos"
    outdict["source_sentence"] = "dummy"
    outdict["target_sentence"] = item['content']
    outdict["source"] = 'Amazon'
    outdict["teacher_encoding"] = [1, 0] if float(item['score']) < 3 else [0, 1]
    outdict["teacher_name"] = "ground_truth"
    outdict["target_language"] = 'zh'
    file_save_dir = f"{save_dir}/{data_role}/Amazon/zh/{class_name}"
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir)
    with open(f"{file_save_dir}/{filename}.json", "w", encoding='utf-8') as outfile:
        json.dump(outdict, outfile)# , ensure_ascii=True)


def preprocess_amazon(data_dir, save_dir, train_langs=['en', 'de'], val_langs=['fr'], test_langs=['jp']):
    langs_split_dict = {}

    for lang in train_langs:
        langs_split_dict[lang] = 'train'
    for lang in val_langs:
        langs_split_dict[lang] = 'val'
    for lang in test_langs:
        langs_split_dict[lang] = 'test'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for sub_dir in os.listdir(data_dir):
        if not sub_dir.startswith('.') and not os.path.isfile(os.path.join(data_dir, sub_dir)):
            lang = sub_dir
            lang_dir = os.path.join(data_dir, lang)
            samples_cnt = 0
            for sub_dir in os.listdir(lang_dir):
                if not sub_dir.startswith('.') and not os.path.isfile(os.path.join(data_dir, sub_dir)):
                    cat = sub_dir
                    cat_dir = os.path.join(lang_dir, cat)
                    for data_file_name in ['train.review', 'test.review']:
                        for event, elem in iterparse(cat_dir + '/' + data_file_name):
                            if elem.tag == 'item':
                                save_sample_amazon(elem, save_dir, lang, langs_split_dict, f'sample_{samples_cnt}')
                                samples_cnt += 1


def preprocess_jd(data_dir, save_dir, data_role='train', subsample=22000):
    infile = codecs.open(data_dir + '/comment.data', 'rb', 'gb2312')
    reviews = {}

    i = 0
    while True:
        try:
            line = infile.readline()
            if line == '':
                break
            score = re.sub('(<score>)|(</score>)', '', re.search('<score>.*</score>', line).group(0))
            content = re.sub('(<content>)|(</content>)', '', re.search('<content>.*</content>', line).group(0))
            reviews[i] = {}
            reviews[i]['score'] = score
            reviews[i]['content'] = content
            i += 1
        except:
            continue

    if subsample:
        samples = random.sample(list(reviews.keys()), k=subsample)
    else:
        samples = list(reviews.keys())
    
    samples_cnt = 0
    for sample in samples:
        save_sample_jd(reviews[sample], save_dir, data_role, f'sample_{samples_cnt}')
        samples_cnt += 1


def ranker_parse_args():
    parser = argparse.ArgumentParser(description='Amazon and JD data preprocessing')
    parser.add_argument('--data_dir_amazon', type=str, default='../../Datasets/amazon_sentiment_polarity/cls-acl10-unprocessed',
                        help='location of the unprocessed Amazon dataset')
    parser.add_argument('--data_dir_jd', type=str, default='../../Datasets/jd', help='location of the unprocessed JD dataset')
    parser.add_argument('--save_dir', type=str, default='../datasets/Dataset', help='directory where processed Amazon dataset is saved')
    parser.add_argument('--train_langs_amazon', type=eval, default='["en", "de"]', help='list of meta-train languages of Amazon dataset')
    parser.add_argument('--val_langs_amazon', type=eval, default='["fr"]', help='list of meta-val languages of Amazon dataset')
    parser.add_argument('--test_langs_amazon', type=eval, default='["jp"]', help='list of meta-test languages of Amazon dataset')
    parser.add_argument('--data_role_jd', type=str, default='train', choices=['train', 'val', 'test'], help='role of the JD Chinese reviews, whether it is used for meta-train, meta-val or meta-test')
    parser.add_argument('--subsample_jd', type=int, default=22000, help='number of samples to take when subsampling, if None get all samples')
    return parser.parse_args()

if __name__ == '__main__':

    args = ranker_parse_args()

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

    preprocess_amazon(args.data_dir_amazon, args.save_dir, args.train_langs_amazon, args.val_langs_amazon, args.test_langs_amazon)
    preprocess_jd(args.data_dir_jd, args.save_dir, args.data_role_jd , args.subsample_jd)

