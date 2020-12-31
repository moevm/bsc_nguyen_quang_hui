# coding=utf-8

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

from common_functions import calculateScore, calculateScoreInterpolation

start = datetime.now()

# several steps to prepare the dataset to be feed to the neural network
# result of this script is 2 file:
# - training set (rebalanced the number of positive and negative sample)
# - test set (25% of original set)

# loading models
nlp = stanza.Pipeline(
    lang='ru', processors='tokenize,pos,lemma')  # ,depparse,ner
NLP_MODEL="xlm-r-bert-base-nli-stsb-mean-tokens"
sent_transformer_model = SentenceTransformer(
    NLP_MODEL)
sent_transformer_model.max_seq_length = 200
# distiluse-base-multilingual-cased: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. While the original mUSE model only supports 16 languages, this multilingual knowledge distilled version supports 50+ languages.
# xlm-r-distilroberta-base-paraphrase-v1 - Multilingual version of distilroberta-base-paraphrase-v1, trained on parallel data for 50+ languages.
# xlm-r-bert-base-nli-stsb-mean-tokens: Produces similar embeddings as the bert-base-nli-stsb-mean-token model. Trained on parallel data for 50+ languages.
# distilbert-multilingual-nli-stsb-quora-ranking - Multilingual version of distilbert-base-nli-stsb-quora-ranking. Fine-tuned with parallel data for 50+ languages.
print("loading model: "+str(datetime.now()-start))
start = datetime.now()


def loadFile(nlp, sent_transformer_model, data_obj, DF={}):
    # search for keyword list

    # print(data_obj)
    keywords = dict()
    for s in data_obj['keywords']:
        if(len(s.strip()) > 0):
            s_temp = nlp(s.strip())
            if(len(s_temp.sentences) > 0):
                kw_lemma = " ".join([
                    w.lemma for w in s_temp.sentences[0].words
                ]).strip()
                keywords[kw_lemma] = {
                    'phrase': s.strip(), 'count': 0, 'first_encounter': -100}

    # tokenizing, lemmarization, ...
    paragraphs = [nlp(p) for p in data_obj['fulltext']]
    # print(paragraphs)
    # print(data_obj)

    # search for keyword candidates in text according to predefined patterns
    keyword_candidates = dict()
    pos_patterns = [
            ["NOUN"], # 789
            # ["ADJ"], # 14
            ["PROPN"], # 589
            ["ADJ", "NOUN"], # 944
            # ["NOUN", "PROPN"], # 10
            ["VERB", "NOUN"], # 49
            # ["NOUN", "PUNCT", "NOUN"], # 43
            ["NOUN", "NOUN"], # 444
            ["PROPN", "PROPN"], # 932
            ["ADJ", "ADJ", "NOUN"], # 85
            ["NOUN", "ADJ", "NOUN"], # 102
            ["NOUN", "NOUN", "NOUN"], # 83
            ["ADJ", "NOUN", "NOUN"], # 71
            ["PROPN", "PROPN", "PROPN"], # 254
            # ["PROPN", "X", "PROPN"], # 47
            # ["PROPN", "X", "PROPN", "PROPN"], # 64
            # ["PROPN", "PUNCT", "PROPN", "PROPN"], # 31
            # ["PROPN", "PROPN", "X", "PROPN"], # 27
            # ["ADJ", "PUNCT", "ADJ", "NOUN"], # 46
            # ["ADJ", "NOUN", "NOUN", "NOUN"], # 10
            # ["NOUN", "NOUN", "ADJ", "NOUN"], # 16
            # ["NOUN", "NOUN", "NOUN", "NOUN"], # 20
            # ["PROPN", "PROPN", "PROPN", "PROPN"], # 34
    ]
    w_count = 1
    for paragraph in paragraphs:
        for sentence in paragraph.sentences:
            for start_index, word in enumerate(sentence.words):
                w_count += 1
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
                            if (lemma_group not in keywords and lemma_group not in keyword_candidates) or (lemma_group in keywords and keywords[lemma_group]['first_encounter'] < 0):
                                try:
                                    DF[lemma_group] += 1
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
                                if 'first_encounter' in keywords[lemma_group] and keywords[lemma_group]['first_encounter'] < 0:
                                    keywords[lemma_group]['first_encounter'] = w_count
                                # increase the count (keyword's encounter in this document)
                                keywords[lemma_group]['count'] += 1

    # print(data_obj)
    part_embeddings = sent_transformer_model.encode(data_obj['fulltext'])

    # print(data_obj)
    kw_lemmas = list(keywords.keys())
    kw_vals = [keywords[lemma]
               for lemma in kw_lemmas]  # list(keywords.values())
    keywords = [k['phrase'] for k in kw_vals]
    keywords_counts = [float(k['count'])/w_count for k in kw_vals]
    keywords_first_encounters = [(float(k['first_encounter']) if float(
        k['first_encounter']) >= 0 else w_count)/w_count for k in kw_vals]
    keyword_embeddings = sent_transformer_model.encode(keywords)
    # calculate avg_cosine and max_cosine for both keywords and phrases

    kc_lemmas = list(keyword_candidates.keys())
    kc_vals = [keyword_candidates[lemma] for lemma in kc_lemmas]
    phrases = [k['phrase'] for k in kc_vals]
    phrases_counts = [float(k['count'])/w_count for k in kc_vals]
    phrases_first_encounters = [(float(k['first_encounter']) if float(
        k['first_encounter']) >= 0 else w_count)/w_count for k in kc_vals]
    phrase_embeddings = sent_transformer_model.encode(phrases)

    # print(part_embeddings)
    return {
        'fname': data_obj['title'],
        # 'text': text,

        'parts': data_obj['fulltext'],
        'part_embeddings': part_embeddings,

        'w_count': w_count,

        'keywords': keywords,
        'keywords_lemmas': kw_lemmas,
        'keywords_lengths': [len(k.split(' '))/4.0 for k in keywords],
        'keywords_counts': keywords_counts,
        'keywords_first_encounters': keywords_first_encounters,
        'keyword_embeddings': keyword_embeddings,

        'phrases': phrases,
        'phrases_lemmas': kc_lemmas,
        'phrases_lengths': [len(k.split(' '))/4.0 for k in phrases],
        'phrases_counts': phrases_counts,
        'phrases_first_encounters': phrases_first_encounters,
        'phrase_embeddings': phrase_embeddings,
    }, DF

DATASET='mixed'
with open("./cyberleninka_"+DATASET+".txt", encoding='utf-8') as fp:
    datafiles = json.load(fp)

datafiles = [d for d in datafiles if d is not None]
print(len(datafiles))
datafiles = datafiles[0:600]
count = 0
keywords_set = []
phrases_set = []
DF = {}
for i in range(len(datafiles)):
    data, DF = loadFile(nlp, sent_transformer_model, datafiles[i], DF)
    # datafiles.append(data)
    if len(data['part_embeddings']) == 0:
        print(data)
        datafiles[i] = None
        print('skipped')
        continue
    count += 1
    datafiles[i] = data
    print(str(count)+". loaded file: " +
          datafiles[i]['fname']+"  \t  ->  "+str(datetime.now()-start))
    start = datetime.now()

datafiles = [d for d in datafiles if d is not None]
print(len(datafiles))

for key in DF:
    DF[key] = float(DF[key])/len(datafiles)
with open('DF.txt', 'w', encoding="utf-8") as outfile:
    json.dump(DF, outfile, ensure_ascii=False)
print("DF construction finished")

for data in datafiles:
    for i in range(len(data['keywords'])):
        avg_score, max_score, max_score_index, five_points = calculateScoreInterpolation(
            data['part_embeddings'], data['keyword_embeddings'][i])
        inp = [
            data['keywords_lengths'][i],
            data['keywords_counts'][i],
            data['keywords_first_encounters'][i],
            avg_score,
            max_score,
            max_score_index,
            DF[data['keywords_lemmas'][i]] if data['keywords_lemmas'][i] in DF else 0
        ] + five_points
        # inp = five_points
        out = [1]
        keywords_set.append({
            'inp': inp,
            'out': out
        })
    for i in range(len(data['phrases'])):
        avg_score, max_score, max_score_index, five_points = calculateScoreInterpolation(
            data['part_embeddings'], data['phrase_embeddings'][i])
        inp = [
            data['phrases_lengths'][i],
            data['phrases_counts'][i],
            data['phrases_first_encounters'][i],
            avg_score,
            max_score,
            max_score_index,
            DF[data['phrases_lemmas'][i]] if data['phrases_lemmas'][i] in DF else 0
        ] + five_points
        # inp = five_points
        out = [0]
        phrases_set.append({
            'inp': inp,
            'out': out
        })

training_keywords = keywords_set[:int(len(keywords_set)*0.75)]
test_keywords = keywords_set[int(len(keywords_set)*0.75):]

training_phrases = phrases_set[:int(len(phrases_set)*0.75)]
test_phrases = phrases_set[int(len(phrases_set)*0.75):]


training_set = []
training_set = training_set + training_phrases
for data in range(int(len(training_phrases)/len(training_keywords))):
    training_set = training_set + training_keywords

print(len(training_set))
with open('training-cl-50f-tfidf-'+DATASET+'.txt', 'w') as outfile:
    json.dump(training_set, outfile)

test_set = test_keywords + test_phrases

print(len(test_set))
with open('test-cl-50f-tfidf-'+DATASET+'.txt', 'w') as outfile:
    json.dump(test_set, outfile)
