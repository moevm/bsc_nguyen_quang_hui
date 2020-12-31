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
    lang='en', processors='tokenize,pos,lemma')  # ,depparse,ner
keyword_patterns = {}
def loadFile(nlp, data_obj, DF):
    # search for keyword list
    # print(data_obj)
    keyword_count = 0
    missing_in_cand = 0
    missing_in_doc = 0
    token_count=0
    candidate_count=0
    keywords = dict()
    kw_pos_patterns = []
    kw_pos_patterns_str = []
    # data_obj['keywords']=[kw for kw in data_obj['keywords'] if len(re.sub(r'[a-z\s]*', '', kw, flags=re.MULTILINE))>1]
    for s in data_obj['keywords']:
        if(len(s.strip()) > 0):
            s_temp = nlp(s.strip())
            if(len(s_temp.sentences) > 0):
                kw_lemma = " ".join([
                    w.lemma for w in s_temp.sentences[0].words
                ]).strip()
                pattern_lst = [w.upos for w in s_temp.sentences[0].words]
                pattern_str = " ".join(pattern_lst)
                if pattern_str not in kw_pos_patterns_str:
                    kw_pos_patterns_str.append(pattern_str)
                    kw_pos_patterns.append(pattern_lst)
                print(pattern_str)
                if pattern_str not in keyword_patterns:
                    keyword_patterns[pattern_str]=1
                else:
                    keyword_patterns[pattern_str]=keyword_patterns[pattern_str]+1

                keywords[kw_lemma] = {'phrase': s.strip(), 'count': 0, 'first_encounter':-100}

    # tokenizing, lemmarization, ...
    paragraphs = [nlp(p) for p in data_obj['fulltext']]
    # print(paragraphs)
    # print(data_obj)

    # finding keywords in document for statistic
    for paragraph in paragraphs:
        for sentence in paragraph.sentences:
            token_count+=len(sentence.words)
            for start_index, word in enumerate(sentence.words):
                for pattern in kw_pos_patterns:
                    if len(sentence.words) >= start_index+len(pattern):
                        matched = True
                        for i, tag in enumerate(pattern):
                            if sentence.words[start_index+i].upos != tag:
                                matched = False
                                break
                        if matched:
                            lemma_group = " ".join(
                                [
                                    w.lemma
                                    for w in sentence.words[start_index:start_index+len(pattern)]
                                ]).strip()
                            if lemma_group in keywords:
                                # if first_encounter is not set 
                                if 'first_encounter' in keywords[lemma_group] and keywords[lemma_group]['first_encounter']==-100:
                                    keywords[lemma_group]['first_encounter'] += 1 
    missing_in_doc+=len([k for k in keywords if keywords[k]['first_encounter']==-100])
    
    keyword_count+=len(keywords)
    # candidate_count+=len(keyword_candidates)+len([k for k in keywords if keywords[k]['first_encounter']>0])
    # missing_in_cand+=len([k for k in keywords if keywords[k]['first_encounter']<0])
    return 1, DF, keyword_count, missing_in_cand, missing_in_doc, token_count, candidate_count


DATASET='inspec'
with open("./dataset_"+DATASET+".txt", encoding='utf-8') as fp:
    datafiles = json.load(fp)

datafiles = [d for d in datafiles if d is not None]
print(len(datafiles))
# datafiles = datafiles[0:600]
count = 0
DF = {}
keyword_count = 0
keyword_missing_in_cand = 0
keyword_missing_in_doc = 0
token_count=0
candidate_count=0
min_token=999999
max_token=0
for i in range(len(datafiles)):
    count+=1
    data, DF, kw_count, missing_in_cand, missing_in_doc, tk_count, cand_count = loadFile(nlp, datafiles[i], DF)
    if max_token<tk_count:
        max_token=tk_count
    if min_token>tk_count:
        min_token=tk_count
    token_count+=tk_count
    keyword_count+=kw_count
    keyword_missing_in_cand+=missing_in_cand
    keyword_missing_in_doc+=missing_in_doc
    candidate_count+=cand_count
    
    print(str(count))
    print(str(keyword_count)+"____"+str(keyword_missing_in_doc)+"____"+str(keyword_missing_in_cand)+"___"+str(token_count)
                +"____"+str(candidate_count)+"____"+str(max_token)+"____"+str(min_token))

print("Keyword count:"+str(keyword_count))
print("Keyword missing in documents:"+str(keyword_missing_in_doc))
print("Keyword missing in candidates:"+str(keyword_missing_in_cand))
print("Number of token:"+str(token_count))
print("Number of candidates:"+str(candidate_count))
print("Max number of token:"+str(max_token))
print("Min number of token:"+str(min_token))
print("Total documents:"+str(count))

# for key in DF:
#     DF[key] = float(DF[key])/len(datafiles)
# with open('DF.txt','w', encoding="utf-8") as outfile:
#     json.dump(DF, outfile, ensure_ascii=False)

with open('keyword_patterns_'+DATASET+'.txt','w', encoding="utf-8") as outfile:
    json.dump(keyword_patterns, outfile, ensure_ascii=False)
# for key in keyword_patterns:
#     print(key+"  "+str(keyword_patterns[key]))
# print("DF construction finished")