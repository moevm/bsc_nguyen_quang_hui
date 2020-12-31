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

from common_functions import calculateScoreInterpolation
start = datetime.now()

# several steps to prepare the dataset to be feed to the neural network
# result of this script is 2 file:
# - training set (rebalanced the number of positive and negative sample)
# - test set (25% of original set)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# loading models
nlp = stanza.Pipeline(
    lang='ru', processors='tokenize,pos,lemma')  # ,depparse,ner
sent_transformer_model = SentenceTransformer(
    'xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
# sent_transformer_model.max_seq_length = 400
# 'distiluse-base-multilingual-cased' , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'distilbert-multilingual-nli-stsb-quora-ranking'

D_in = 29  # kw_length, kw_count, avg_cosine, max_cosine
# H1 = 90
# H2 = 180
# H3 = 90
D_out = 1
H = 450
kw_assessment_model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H, H),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)

kw_assessment_model.to(device)
kw_assessment_model.load_state_dict(torch.load("./state_dict.pt"))

kw_assessment_model.eval()

print("loading model: "+str(datetime.now()-start))
start = datetime.now()


def load_course_article(filename):
    f = open(filename, 'r', encoding='utf-8')
    text = f.read()
    f.close()

    # search for keyword list
    keywordSearchRegex = r"(?<=Ключевые слова: )[\S\s]*?(?=(\n))"
    keywordReplaceRegex = r"(Ключевые слова: )[\S\s]*?(?=(\n))"
    search = re.search(keywordSearchRegex, text)
    if(search is None):
        return None
    keywords = search.group().replace(';', ',').replace('.', '').split(',')

    text = re.sub(keywordReplaceRegex, "", text)

    text_parts = [s for s in re.findall(
        r"((?<=#+[\s\S]+?\n)[\s\S]*?(?=#)|$)", text) if len(s) > 10]

    text = text.replace("#", "").replace("*", "")
    title = text.splitlines()[0]
    paragraphs = [s.strip() for s in text.splitlines() if len(s.strip()) > 150]
    return {
        'title': title,
        'keywords': keywords,
        'fulltext': paragraphs
    }


def loadFile(nlp, sent_transformer_model, data_obj):
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
                            if lemma_group not in keywords:
                                if lemma_group not in keyword_candidates:
                                    keyword_candidates[lemma_group] = {
                                        'phrase': phrase,
                                        'count': 1,
                                        'first_encounter': w_count
                                    }
                                else:
                                    keyword_candidates[lemma_group]['count'] += 1
                            else:
                                if 'first_encounter' in keywords[lemma_group] and keywords[lemma_group]['first_encounter'] < 0:
                                    keywords[lemma_group]['first_encounter'] = w_count
                                keywords[lemma_group]['count'] += 1

    # print(data_obj)
    part_embeddings = sent_transformer_model.encode(data_obj['fulltext'])
    # print(data_obj)
    kw_lemmas = list(keywords.keys())
    kw_vals = [keywords[lemma] for lemma in kw_lemmas] #list(keywords.values())
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
    }


# def calculateScore(parts_embeddings, keyword_embeddings):
#     avg_score = 0
#     max_score = 0
#     max_score_index = 0
#     index = 0
#     for part_emb in parts_embeddings:
#         score = torch.nn.functional.cosine_similarity(
#             torch.tensor(part_emb), torch.tensor(keyword_embeddings), dim=-1)
#         # print(score)
#         avg_score += score.item()
#         if(score.item() > max_score):
#             max_score = score.item()
#             max_score_index = index
#         index += 1
#     avg_score = (avg_score/len(parts_embeddings)+1)/2
#     max_score = (max_score+1)/2
#     max_score_index = max_score_index / len(parts_embeddings)
#     return avg_score, max_score, max_score_index


# with open("./cyberleninka_ds.txt", encoding='utf-8') as fp:
#     datafiles = json.load(fp)

# datafiles = [d for d in datafiles if d is not None]

# # article number 15 in cyberleninka dataset
# file = datafiles[15]
with open("./DF.txt", encoding='utf-8') as fp:
    DF = json.load(fp)
article_numbers = [8]#[x for x in range(3,41)]#[8,11,29,31,32,33,34,35,39]
for no in article_numbers:
    # article from sw course
    file_name = "paper2 ("+str(no)+")"
    file = load_course_article("./dataset/"+file_name+".md")

    data = loadFile(nlp, sent_transformer_model, file)


    # datafiles.append(data)
    if len(data['part_embeddings']) == 0:
        print(data)
        print('skipped')
        exit(0)

    # count+=1
    print("loaded file: "+file['title']+"  \t  ->  "+str(datetime.now()-start))
    start = datetime.now()

    keywords_set = []
    phrases_set = []

    for i in range(len(data['keywords'])):
        avg_score, max_score, max_score_index, five_points = calculateScoreInterpolation(
            data['part_embeddings'], data['keyword_embeddings'][i])
        inp = [
            data['keywords_lengths'][i],
            data['keywords_counts'][i],
            data['keywords_first_encounters'][i],
            # avg_score,
            # max_score,
            # max_score_index,
            DF[data['keywords_lemmas'][i]] if data['keywords_lemmas'][i] in DF else 0
        ]+five_points
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
            # avg_score,
            # max_score,
            # max_score_index,
            DF[data['phrases_lemmas'][i]] if data['phrases_lemmas'][i] in DF else 0
        ]+five_points
        out = [0]
        phrases_set.append({
            'inp': inp,
            'out': out
        })

    with open("result_"+file_name+".txt", "w", encoding="utf-8") as f:
        print(data['fname'])
        print(data['parts'])
        f.write("Article name: \n\t"+ str(data['fname'])+"\n")

        f.write("Parts:\n")
        for part in data['parts']:
            f.write("\tlen("+str(len(part))+"): "+part+"\n")
        with torch.no_grad():
            print("------------------- KEYWORDS")
            f.write("------------------- KEYWORDS\n")
            for index, kw in enumerate(keywords_set):
                # extract a batch from dataset
                x = torch.tensor(kw['inp'], device=device)
                kw_assessment_model.eval()
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = kw_assessment_model(x)
                y_bin = (y_pred > 0.5).float()

                print(data['keywords'][index]+"   "+str(y_pred) +
                    "   "+str(y_bin.item() == kw['out'][0]))
                f.write(data['keywords'][index]+"   "+str(y_pred) +
                    "   "+str(y_bin.item() == kw['out'][0])+"\n")

            print("------------------- PHRASES")
            f.write("------------------- PHRASES\n")
            for index, kw in enumerate(phrases_set):
                # extract a batch from dataset
                x = torch.tensor(kw['inp'], device=device)
                kw_assessment_model.eval()
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = kw_assessment_model(x)
                y_bin = (y_pred > 0.5).float()

                if not y_bin.item() == kw['out'][0]:
                    print(data['phrases'][index]+"   "+str(y_pred.item()) +
                        "   "+str(y_bin.item() == kw['out'][0]))
                    f.write(data['phrases'][index]+"   "+str(y_pred.item()) +
                        "   "+str(y_bin.item() == kw['out'][0])+"\n")
