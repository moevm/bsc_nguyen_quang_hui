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

# several steps to prepare the dataset to be feed to the neural network
# result of this script is 2 file:
# - training set (rebalanced the number of positive and negative sample)
# - test set (25% of original set)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# loading models
nlp = stanza.Pipeline(
    lang='en', processors='tokenize,pos,lemma')  # ,depparse,ner
sent_transformer_model = SentenceTransformer(
    'xlm-r-bert-base-nli-stsb-mean-tokens')
sent_transformer_model.max_seq_length = 200
# 'distiluse-base-multilingual-cased' , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'distilbert-multilingual-nli-stsb-quora-ranking'

D_in = 29  # kw_length, kw_count, avg_cosine, max_cosine
D_out = 1
H = 1200
kw_assessment_model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.LeakyReLU(),
    # torch.nn.Dropout(0.2),
    # torch.nn.Linear(H, H),
    # torch.nn.LeakyReLU(),
    # torch.nn.Dropout(0.2),
    # torch.nn.Linear(H, H),
    # torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)

kw_assessment_model.to(device)
kw_assessment_model.load_state_dict(torch.load("./state_dict.pt"))

kw_assessment_model.eval()

print("Model loaded")

class KeywordExtractor:
    
    # load the content of the document, here document is expected to be in raw
    # format (i.e. a simple text file) and preprocessing is carried out using spacy
    def __init__(self):
        self.kw_assessment_model = kw_assessment_model
        self.nlp = nlp
        self.sent_transformer_model = sent_transformer_model
        
    def load_document(self, input, language):
        self.fulltext = input.split('\n')

        # tokenizing, lemmarization, ...
        self.paragraphs = [self.nlp(p) for p in self.fulltext]
    
    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    def candidate_selection(self):
        # search for keyword candidates in text according to predefined patterns
        keyword_candidates = dict()
        pos_patterns = [
            ["NOUN"],
            ["PROPN"],
            ['ADJ'],
            ['VERB', 'NOUN'],
            ["ADJ", "NOUN"],
            ["NOUN", "NOUN"],
            ["NOUN", "PROPN"],
            ["ADJ", "ADJ", "NOUN"],
            ["NOUN", "NOUN", "NOUN"],
            ["ADJ", "NOUN", "NOUN"]
        ]
        w_count = 1
        for paragraph in self.paragraphs:
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
                                if lemma_group not in keyword_candidates:
                                    keyword_candidates[lemma_group] = {
                                        'phrase': phrase,
                                        'count': 1,
                                        'first_encounter': w_count
                                    }
                                else:
                                    keyword_candidates[lemma_group]['count'] += 1

        self.part_embeddings = self.sent_transformer_model.encode(self.fulltext)

        self.kc_lemmas = list(keyword_candidates.keys())
        self.kc_vals = [keyword_candidates[lemma] for lemma in self.kc_lemmas]
        self.phrases = [k['phrase'] for k in self.kc_vals]
        self.phrases_counts = [float(k['count'])/w_count for k in self.kc_vals]
        self.phrases_lengths = [len(k.split(' '))/4.0 for k in self.phrases]
        self.phrases_first_encounters = [(float(k['first_encounter']) if float(
            k['first_encounter']) >= 0 else w_count)/w_count for k in self.kc_vals]
        self.phrase_embeddings = self.sent_transformer_model.encode(self.phrases)

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    def candidate_weighting(self,df={}):
        self.phrases_set = []
        for i in range(len(self.phrases)):
            avg_score, max_score, max_score_index, five_points = calculateScoreInterpolation(
                self.part_embeddings, self.phrase_embeddings[i])
            inp = [
                self.phrases_lengths[i],
                self.phrases_counts[i],
                self.phrases_first_encounters[i],
                df[self.kc_lemmas[i]] if self.kc_lemmas[i] in df else 0.006
            ]+five_points
            # extract a batch from dataset
            x = torch.tensor(inp, device=device)
            self.kw_assessment_model.eval()
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.kw_assessment_model(x)
            y_bin = (y_pred > 0.5).float()
            self.phrases_set.append({
                'phrase': self.phrases[i],
                'features': inp,
                'score': y_pred.item()
            })
        self.phrases_set.sort(key=lambda x: x['score'], reverse=True)

    def get_n_best(self, n=10):
        if n<10000:
            result = [(phrase['phrase'], phrase['score']) for phrase in self.phrases_set[:n]] # if phrase['score']>0.5]
        else:
            result = [(phrase['phrase'], phrase['score']) for phrase in self.phrases_set]
        return result

