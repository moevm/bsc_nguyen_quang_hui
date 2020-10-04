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

from common_functions import calculateScore 


start = datetime.now()

# take in an article and a keyword, return a score indicate how well the keyword is



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# loading models
nlp = stanza.Pipeline(
    lang='ru', processors='tokenize,pos,lemma')  # ,depparse,ner
sent_transformer_model = SentenceTransformer(
    'xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
# sent_transformer_model.max_seq_length = 400
# 'distiluse-base-multilingual-cased' , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'distilbert-multilingual-nli-stsb-quora-ranking'

# TODO: load pytorch model
# kw_assessment_model = torch.load("./latest.model", map_location=device)
# kw_assessment_model.to(device)

D_in = 6  # kw_length, kw_count, avg_cosine, max_cosine
H1 = 32
H2 = 64
H3 = 32
D_out = 1

kw_assessment_model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H2, H3),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H3, D_out),
    torch.nn.Sigmoid(),
)

kw_assessment_model.to(device)
kw_assessment_model.load_state_dict(torch.load("./state_dict.pt"))

print(kw_assessment_model)
kw_assessment_model.eval()
for param in kw_assessment_model.parameters():
    print(param.data)

print("loading model: "+str(datetime.now()-start))
start = datetime.now()


def loadFile(nlp, sent_transformer_model, name):
    f = open(name, 'r', encoding='utf-8')
    text = f.read()
    f.close()

    # search for keyword list
    keywordSearchRegex = r"(?<=Ключевые слова: )[\S\s]*?(?=(\n))"
    keywordReplaceRegex = r"(Ключевые слова: )[\S\s]*?(?=(\n))"
    search = re.search(keywordSearchRegex, text)
    keywords = []
    if(search is not None):
        keywords = [s.strip() for s in search.group().replace(';',',').replace('.','').split(',')]

    text = re.sub(keywordReplaceRegex, "", text) # input text with stripped keyword 

    text_parts = [s for s in re.findall(
        r"((?<=#+[\s\S]+?\n)[\s\S]*?(?=#)|$)", text) if len(s) > 10] # input text separated into parts (separated by markdown's headers)

    text = text.replace("#", "").replace("*", "")
    text = os.linesep.join([s.strip() for s in text.splitlines() if s.strip()])
    paragraphs = text.split("\n")

    # tokenizing, lemmarization, ...
    paragraphs = [nlp(p) for p in paragraphs]

    # count words in the text
    w_count = 1
    for paragraph in paragraphs:
        for sentence in paragraph.sentences:
            w_count += len(sentence.words)

    part_embeddings = sent_transformer_model.encode(text_parts)

    return {
        'text': text,
        'keywords': keywords,
        'parts': text_parts,
        'part_embeddings': part_embeddings,

        'w_count': w_count,

        'paragraphs': paragraphs
    }


def preprocess_keyword(nlp, sent_transformer_model, text_info, keyword):
    s_temp = nlp(keyword.strip())
    if(len(s_temp.sentences) > 0):
        kw_lemma = [w.lemma for w in s_temp.sentences[0].words]
    else:
        return None

    count = 0
    first_encounter = -1
    w_count = 0
    for paragraph in text_info['paragraphs']:
        for sentence in paragraph.sentences:
            for start_index, word in enumerate(sentence.words):
                w_count += 1
                if len(sentence.words) >= start_index+len(kw_lemma):
                    matched = True
                    for i, val in enumerate(kw_lemma):
                        if sentence.words[start_index+i].lemma != val:
                            matched = False
                            break
                    if matched:
                        count += 1
                        if first_encounter < 0:
                            first_encounter = w_count

    kw_embeddings = sent_transformer_model.encode([keyword])[0]
    avg_score, max_score, max_score_index = calculateScore(
        text_info['part_embeddings'], kw_embeddings)
    return {
        'length': len(kw_lemma)/4.0,
        'count': float(count)/w_count,
        'first_encounter': float(first_encounter)/w_count if first_encounter>=0 else 1,
        'avg_score': avg_score,
        'max_score': max_score,
        'max_score_index': max_score_index
    }


def assess_keyword(keyword_info):
    inp_tensor = torch.tensor([[
        keyword_info['length'],
        keyword_info['count'],
        keyword_info['first_encounter'],
        keyword_info['avg_score'],
        keyword_info['max_score'],
        keyword_info['max_score_index']
    ]]).to(device).float()
    print(inp_tensor)
    kw_assessment_model.eval()
    prediction = kw_assessment_model(inp_tensor)
    print(prediction)


# read input text from file
text_info = loadFile(nlp, sent_transformer_model, "./dataset/paper2 (30).md")
print("loaded file\t  ->  "+str(datetime.now()-start))
start = datetime.now()

keyword = "причины необходимости создания имитационной платформы SOA-систем"

keyword_info = preprocess_keyword(
    nlp, sent_transformer_model, text_info, keyword)

print("preprocess: "+str(datetime.now()-start))
start = datetime.now()

assess_keyword(keyword_info)

print("assessment: "+str(datetime.now()-start))
start = datetime.now()
