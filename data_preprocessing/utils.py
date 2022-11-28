import os
import re
import json
from html import unescape
from emoji import demojize

def clean_sentence(sentence):
    sentence = re.sub(r"https?://\S+", "", sentence) # remove hyperlinks
    sentence = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", sentence) # remove <a> tags but keep their content
    sentence = re.sub(r"<.*?>", " ", sentence) # remove all HTML tags but keep their contents
    sentence = re.sub(r"\b[0-9]+\b\s*", "", sentence) # remove numbers
    sentence = unescape(sentence) # remove special characters
    sentence = re.sub(r"[#-%+*/@]", "", sentence) # remove special characters
    sentence = re.sub(r'(.)\1{3,}',r'\1', sentence) # remove repeated characters
    sentence = " ".join(sentence.split()) # remove extra spaces, tabs, and line breaks
    sentence = demojize(sentence) # transform emojis into characters
    return sentence

def save_sample_amazon(item, save_dir, lang, langs_split_dict, filename):
    outdict = {}
    class_name = "neg" if float(item.find("rating").text) < 3 else "pos"
    outdict["source_sentence"] = "dummy"
    outdict["target_sentence"] = clean_sentence(str(item.find("text").text))
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
    outdict["target_sentence"] = clean_sentence(item['content'])
    outdict["source"] = 'Amazon'
    outdict["teacher_encoding"] = [1, 0] if float(item['score']) < 3 else [0, 1]
    outdict["teacher_name"] = "ground_truth"
    outdict["target_language"] = 'zh'
    file_save_dir = f"{save_dir}/{data_role}/Amazon/zh/{class_name}"
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir)
    with open(f"{file_save_dir}/{filename}.json", "w", encoding='utf-8') as outfile:
        json.dump(outdict, outfile)# , ensure_ascii=True)

def save_sample_covid(item, save_dir, lang, langs_split_dict, filename):
    outdict = {}
    class_name = item['label']
    outdict["source_sentence"] = "dummy"
    outdict["target_sentence"] = clean_sentence(item['ref_source']['text'])
    outdict["source"] = 'MM-COVID'
    outdict["teacher_encoding"] = [1, 0] if item['label'] == 'fake' else [0, 1]
    outdict["teacher_name"] = "ground_truth"
    outdict["target_language"] = lang
    file_save_dir = f"{save_dir}/{langs_split_dict[lang]}/MM-COVID/{lang}/{class_name}"
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir)
    with open(f"{file_save_dir}/{filename}.json", "w", encoding='utf-8') as outfile:
        json.dump(outdict, outfile)# , ensure_ascii=True)