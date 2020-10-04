# coding=utf-8

## THIS IS FOR DATASOURCE IN DATASET FOLDER, EACH MD FILE CONTAINS AN ARTICLE
from os import listdir
from os.path import isfile, join
import stanza
import regex as re
import os
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset, DataLoader
import json


from common_functions import calculateScore
start = datetime.now()

# loading models
nlp = stanza.Pipeline(
    lang='ru', processors='tokenize,pos,lemma')  # ,depparse,ner
sent_transformer_model = SentenceTransformer(
    'xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
# sent_transformer_model.max_seq_length = 400
# 'distiluse-base-multilingual-cased' , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'distilbert-multilingual-nli-stsb-quora-ranking'
print("loading model: "+str(datetime.now()-start))
start = datetime.now()


def loadFile(nlp, sent_transformer_model, name, DF = {}):
    f = open(name, 'r', encoding='utf-8')
    text = f.read()
    f.close()

    # search for keyword list
    keywordSearchRegex = r"(?<=Ключевые слова: )[\S\s]*?(?=(\n))"
    keywordReplaceRegex = r"(Ключевые слова: )[\S\s]*?(?=(\n))"
    search = re.search(keywordSearchRegex, text)
    if(search is None):
        return None
    keywords = dict()
    for s in search.group().replace(';',',').replace('.','').split(','):
        if(len(s.strip()) > 0):
            s_temp = nlp(s.strip())
            if(len(s_temp.sentences) > 0):
                kw_lemma = " ".join([
                    w.lemma for w in s_temp.sentences[0].words
                ]).strip()
                keywords[kw_lemma] = {'phrase': s.strip(), 'count': 0, 'first_encounter':-100}

    # print(keywords.encode('utf-8'))

    text = re.sub(keywordReplaceRegex, "", text)

    text_parts = [s for s in re.findall(
        r"((?<=#+[\s\S]+?\n)[\s\S]*?(?=#)|$)", text) if len(s) > 10]

    text = text.replace("#", "").replace("*", "")
    text = os.linesep.join([s.strip() for s in text.splitlines() if s.strip()])
    paragraphs = text.split("\n")
    # print(text)

    # tokenizing, lemmarization, ...
    paragraphs = [nlp(p) for p in paragraphs]
    # print(paragraphs)

    # search for keyword candidates in text according to predefined patterns
    keyword_candidates = dict()
    pos_patterns = [
        ["NOUN"],
        ["PROPN"],
        ["PROPN", "PROPN"],
        ["ADJ", "NOUN"],
        ["NOUN", "NOUN"],
        ["NOUN", "PROPN"],
        ["NOUN", "PROPN", "PROPN"],
        ["ADJ", "NOUN", "PROPN"],
        ["ADJ", "ADJ", "NOUN"],
        ["NOUN", "NOUN", "NOUN"],
        ["ADJ", "NOUN", "NOUN"],
    ]
    w_count = 0
    for paragraph in paragraphs:
        for sentence in paragraph.sentences:
            for start_index, word in enumerate(sentence.words):
                w_count+=1
                for pattern in pos_patterns:
                    if len(sentence.words) >= start_index+len(pattern):
                        matched = True
                        for i, tag in enumerate(pattern):
                            if sentence.words[start_index+i].upos != tag:
                                matched = False
                                break
                        if matched:
                            phrase = " ".join(
                                [
                                    w.text.lower()
                                    for w in sentence.words[start_index:start_index+len(pattern)]
                                ]).strip()
                            lemma_group = " ".join(
                                [
                                    w.lemma
                                    for w in sentence.words[start_index:start_index+len(pattern)]
                                ]).strip()
                            if (lemma_group not in keywords and lemma_group not in keyword_candidates) or (lemma_group in keywords and keywords[lemma_group]['first_encounter']<0):
                                try:
                                    DF[lemma_group]+=1
                                except:
                                    DF[lemma_group] = 1
                            if lemma_group not in keywords:
                                # this lemma_group is not in the author defined keyword list, put it in the keyword_candidates (non-keyword phrase list)
                                if lemma_group not in keyword_candidates:
                                    keyword_candidates[lemma_group] = {
                                        'phrase': phrase,
                                        'count': 1,
                                        'first_encounter': w_count
                                    }
                                else:
                                    keyword_candidates[lemma_group]['count'] += 1
                            else:
                                # if first_encounter is not set (default to <0) then set it
                                if keywords[lemma_group]['first_encounter']<0:
                                    keywords[lemma_group]['first_encounter'] = w_count 
                                # increase the count (keyword's encounter in this document)
                                keywords[lemma_group]['count'] += 1

    part_embeddings = sent_transformer_model.encode(text_parts)

    kw_vals = list(keywords.values())
    keywords = [k['phrase'] for k in kw_vals]
    keywords_counts = [float(k['count'])/w_count for k in kw_vals]
    keywords_first_encounters = [(float(k['first_encounter']) if float(k['first_encounter'])>=0 else w_count)/w_count for k in kw_vals]
    keyword_embeddings = sent_transformer_model.encode(keywords)
    # calculate avg_cosine and max_cosine for both keywords and phrases

    kc_vals = list(keyword_candidates.values())
    phrases = [k['phrase'] for k in kc_vals]
    phrases_counts = [float(k['count'])/w_count for k in kc_vals]
    phrases_first_encounters = [(float(k['first_encounter']) if float(k['first_encounter'])>=0 else w_count)/w_count for k in kc_vals]
    phrase_embeddings = sent_transformer_model.encode(phrases)

    # print(part_embeddings)
    return {
        'fname': name,
        'text': text,

        'parts': text_parts,
        'part_embeddings': part_embeddings,

        'w_count': w_count,

        'keywords': keywords,
        'keywords_lengths': [len(k.split(' '))/4.0 for k in keywords],
        'keywords_counts': keywords_counts,
        'keywords_first_encounters': keywords_first_encounters,
        'keyword_embeddings': keyword_embeddings,

        'phrases': phrases,
        'phrases_lengths': [len(k.split(' '))/4.0 for k in phrases],
        'phrases_counts': phrases_counts,
        'phrases_first_encounters':phrases_first_encounters,
        'phrase_embeddings': phrase_embeddings,
    }, DF

# read input text from file
files = [join("./dataset", f)
         for f in listdir("./dataset") if isfile(join("./dataset", f))]

# FOR TESTING
files = files[38:]

datafiles = []
DF = {}
for file in files:
    data, DF = loadFile(nlp, sent_transformer_model, file, DF)
    datafiles.append(data)
    print("loaded file: "+file+"  \t  ->  "+str(datetime.now()-start))
    start = datetime.now()


datafiles = [d for d in datafiles if d is not None]
print(len(datafiles))

for key in DF:
    DF[key] = float(DF[key])/len(datafiles)
with open('DF.txt','w') as outfile:
    json.dump(DF, outfile)
print("DF construction finished")


training_files = datafiles[:int(len(datafiles)*0.75)]
test_files = datafiles[int(len(datafiles)*0.75):]
training_set = []
for data in training_files:
    for i in range(len(data['phrases'])):
        avg_score, max_score = calculateScore(
            data['part_embeddings'], data['keyword_embeddings'][i%len(data['keywords'])])
        inp = [
            data['keywords_lengths'][i%len(data['keywords'])],
            data['keywords_counts'][i%len(data['keywords'])],
            data['keywords_first_encounters'][i%len(data['keywords'])],
            avg_score,
            max_score
        ]
        out = [1]
        training_set.append({
            'inp': inp,
            'out': out
        })
        avg_score, max_score = calculateScore(
            data['part_embeddings'], data['phrase_embeddings'][i])
        inp = [
            data['phrases_lengths'][i],
            data['phrases_counts'][i],
            data['phrases_first_encounters'][i],
            avg_score,
            max_score
        ]
        out = [0]
        training_set.append({
            'inp': inp,
            'out': out
        })

print(len(training_set))
with open('training.txt','w') as outfile:
    json.dump(training_set, outfile)

test_set = []
for data in test_files:
    for i in range(len(data['keywords'])):
        avg_score, max_score = calculateScore(
            data['part_embeddings'], data['keyword_embeddings'][i])
        inp = [
            data['keywords_lengths'][i],
            data['keywords_counts'][i],
            data['keywords_first_encounters'][i],
            avg_score,
            max_score
        ]
        out = [1]
        print(data['fname']+"   "+data['keywords'][i]+"    "+str(inp))
        test_set.append({
            'inp': inp,
            'out': out
        })
    for i in range(len(data['phrases'])):
        avg_score, max_score = calculateScore(
            data['part_embeddings'], data['phrase_embeddings'][i])
        inp = [
            data['phrases_lengths'][i],
            data['phrases_counts'][i],
            data['phrases_first_encounters'][i],
            avg_score,
            max_score
        ]
        out = [0]
        test_set.append({
            'inp': inp,
            'out': out
        })

print(len(test_set))
with open('test.txt','w') as outfile:
    json.dump(test_set, outfile)

