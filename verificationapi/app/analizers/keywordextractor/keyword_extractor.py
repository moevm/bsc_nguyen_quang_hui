# coding=utf-8

import time
import torch
import json
from pathlib import Path
import nltk
from nltk.stem.snowball import SnowballStemmer

from .common_functions import calculateScoreInterpolation

from analizers.default_preprocessing_modules import create_default_encoder

from .pos_patterns import POS_PATTERNS

from analizers.abstract_analizer import AbstractAnalizer


class KeywordExtractor(AbstractAnalizer):

    def __init__(self, language='en', nlp_module=None, encoder=None):
        super().__init__(language, nlp_module)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # self.device = torch.device('cpu')

        if encoder is None:
            # loading SBERT model
            self.sent_transformer_model = create_default_encoder()
        else:
            self.sent_transformer_model = encoder
        
        # quantize the encoder model (SBERT) to int8 on cpu and float16 on gpu
        # if not self.use_cuda:
        #     self.sent_transformer_model = torch.quantization.quantize_dynamic(self.sent_transformer_model,{torch.nn.Linear},dtype=torch.qint8)

        print("SBERT model loaded")

        D_in = 29  # kw_length, kw_count, avg_cosine, max_cosine
        D_out = 1
        H = 450
        self.kw_assessment_model = torch.nn.Sequential(
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

        base_path = Path(__file__).parent
        # file_path = (base_path / "../data/test.csv").resolve()
        if language == 'en':
            head_model_filename = (
                base_path / './eng_nus_25_model.pt').resolve()
            df_filename = (base_path / "./DF_eng.txt").resolve()
        elif language == 'ru':
            head_model_filename = (
                base_path / './rus_clmixed_25_model.pt').resolve()
            df_filename = (base_path / "./DF_rus.txt").resolve()
        else:
            raise Exception('Language not supported')

        self.kw_assessment_model.load_state_dict(
            torch.load(head_model_filename, map_location=self.device))

        self.kw_assessment_model.to(self.device)

        self.kw_assessment_model.eval()

        print("Keyword evaluator head model loaded")

        # self.stemmer = SnowballStemmer(language='russian' if language=='ru' else 'english')
        # load df
        self.df = {}
        with open(df_filename, encoding='utf-8') as fp:
            self.df = json.load(fp)

    def execute(self, state, is_evaluator=True, is_extractor=False, user_phrases=[], n=10):
        if not is_evaluator and not is_extractor:
            raise Exception(
                'Must enable at least 1 of 2 modes (is_evaluator or is_extractor)')

        # tokenizing, lemmarization, ...
        paragraphs = [s.strip()
                      for s in state['paragraphs'] if len(s.strip()) > 150]
        print(len(paragraphs))
        start_time = time.time()
        part_embeddings = self.sent_transformer_model.encode(paragraphs, batch_size=32 if self.use_cuda else 1)
        print(time.time()-start_time)
        state['part_embeddings'] = part_embeddings
        state['keyword_candidates'] = dict()
        state['user_phrases'] = dict()

        if is_evaluator:
            self.user_phrases_processing(state, user_phrases)
        if is_extractor:
            self.candidate_selection(state)

        self.candidate_processing(state)

        if is_evaluator and is_extractor:
            return state['user_phrases'], state['keyword_candidates']
        if is_evaluator:
            return state['user_phrases']
        if is_extractor:
            return state['keyword_candidates']

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    def candidate_selection(self, state):
        # search for keyword candidates in text according to predefined patterns
        state['keyword_candidates'] = dict()
        pos_patterns = POS_PATTERNS[state['language']]
        w_count = 1
        for paragraph in state['tokenized_paragraphs']:
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
                                if lemma_group not in state['keyword_candidates']:
                                    state['keyword_candidates'][lemma_group] = {
                                        'phrase': phrase,
                                        'count': 1,
                                        'first_encounter': w_count
                                    }
                                else:
                                    state['keyword_candidates'][lemma_group]['count'] += 1
        state['w_count'] = w_count
        return state

    def user_phrases_processing(self, state, user_phrases):
        # search for keyword candidates in text according to predefined patterns
        state['user_phrases'] = dict()

        # TODO: integrate
        # for s in data_obj['keywords']:
        #     if(len(s.strip()) > 0):
        #         s_temp = nlp(s.strip())
        #         if(len(s_temp.sentences) > 0):
        #             kw_lemma = " ".join([
        #                 w.lemma for w in s_temp.sentences[0].words
        #             ]).strip()
        #             keywords[kw_lemma] = {
        #                 'phrase': s.strip(), 'count': 0, 'first_encounter': -100}

        w_count = 1
        user_phrases = [[w.to_dict() for w in self.nlp(
            p).sentences[0].words] for p in user_phrases]

        for phrase in user_phrases:
            kw_lemma = " ".join([w['lemma'] for w in phrase]).strip()
            kw_text = " ".join([w['text'] for w in phrase]).strip()
            state['user_phrases'][kw_lemma] = {
                'phrase': kw_text, 'count': 0, 'first_encounter': -100}
        # print(user_phrases)
        # for phrase in user_phrases:
        #     for w in phrase:
        #         w['stemming'] = self.stemmer.stem(w['lemma'])
        # print(user_phrases)
        for paragraph in state['tokenized_paragraphs']:
            for sentence in paragraph.sentences:
                for start_index, word in enumerate(sentence.words):
                    w_count += 1
                    for user_phrase in user_phrases:
                        if len(sentence.words) >= start_index+len(user_phrase):
                            matched = True
                            for i, word in enumerate(user_phrase):
                                # print(self.stemmer.stem(sentence.words[start_index+i].lemma), self.stemmer.stem(word.lemma))
                                # if self.stemmer.stem(sentence.words[start_index+i].lemma) != word['stemming']:
                                if sentence.words[start_index+i].lemma != word['lemma']:
                                    matched = False
                                    break
                            if matched:
                                phrase = " ".join(
                                    [
                                        w['text'].lower()
                                        for w in user_phrase
                                    ]).strip()
                                lemma_group = " ".join(
                                    [
                                        w['lemma']
                                        # w['stemming']
                                        for w in user_phrase
                                    ]).strip()
                                if lemma_group not in state['user_phrases']:
                                    state['user_phrases'][lemma_group] = {
                                        'phrase': phrase,
                                        'count': 1,
                                        'first_encounter': w_count
                                    }
                                else:
                                    if 'first_encounter' in state['user_phrases'][lemma_group] and state['user_phrases'][lemma_group]['first_encounter'] < 0:
                                        state['user_phrases'][lemma_group]['first_encounter'] = w_count
                                    state['user_phrases'][lemma_group]['count'] += 1
        state['w_count'] = w_count
        return state

    def candidate_processing(self, state):
        candidates = {**state['user_phrases'], **state['keyword_candidates']}

        def set_score(state, lemma, score):
            if lemma in state['user_phrases']:
                state['user_phrases'][lemma]['score'] = score
                # state['user_phrases'][lemma]['class'] = 'keyword' if y_bin == True else 'non-keyword'
            if lemma in state['keyword_candidates']:
                state['keyword_candidates'][lemma]['score'] = score
                # state['user_phrases'][lemma]['class'] = 'keyword' if y_bin == True else 'non-keyword'

        kc_lemmas = list(candidates.keys())
        kc_vals = [candidates[lemma] for lemma in kc_lemmas]
        phrases = [k['phrase'] for k in kc_vals]
        phrases_counts = [float(k['count'])/state['w_count'] for k in kc_vals]
        phrases_lengths = [len(k.split(' '))/4.0 for k in phrases]
        phrases_first_encounters = [(float(k['first_encounter']) if float(
            k['first_encounter']) >= 0 else state['w_count'])/state['w_count'] for k in kc_vals]

        phrase_embeddings = self.sent_transformer_model.encode(phrases)
        # phrases_set = []
        input_array = []
        for i in range(len(phrases)):
            avg_score, max_score, max_score_index, five_points = calculateScoreInterpolation(
                state['part_embeddings'], phrase_embeddings[i])
            inp = [
                phrases_lengths[i],
                phrases_counts[i],
                phrases_first_encounters[i],
                self.df[kc_lemmas[i]] if kc_lemmas[i] in self.df else 0
            ]+five_points
            input_array.append(inp)

        # batch size when applying the head model
        batch_size = 32

        # add padding to the input array so that it divisible by batch size
        while len(input_array)%batch_size!=0:
            input_array.append([0]*29)
        # feed the input through the head model
        x = torch.tensor(input_array, device=self.device).view(-1,batch_size,29)
        y_pred = self.kw_assessment_model(x).view(-1,1)

        for i in range(len(phrases)):
            # # extract a batch from dataset
            # x = torch.tensor(inp, device=self.device)
            # # Forward pass: compute predicted y by passing x to the model.
            # y_pred = self.kw_assessment_model(x)
            # y_bin = (y_pred > 0.5).float()
            set_score(state, kc_lemmas[i], y_pred[i].item())

        return state

    # def get_n_best(self, state, n=10):
    #     phrases_set = state['keyword_candidates']
    #     phrases_set.sorted(key=lambda x: x['score'], reverse=True)
    #     if n < 10000:
    #         result = [phrase
    #                   for phrase in phrases_set[:n]]  # if phrase['score']>0.5]
    #     else:
    #         result = [phrase
    #                   for phrase in phrases_set]
    #     return result

    # def candidate_processing(self, state):
    #     candidates = {**state['user_phrases'] , **state['keyword_candidates']}
    #     state['kc_lemmas'] = list(candidates.keys())
    #     state['kc_vals'] = [candidates[lemma] for lemma in state['kc_lemmas']]
    #     state['phrases'] = [k['phrase'] for k in state['kc_vals']]
    #     state['phrases_counts'] = [float(k['count'])/state['w_count'] for k in state['kc_vals']]
    #     state['phrases_lengths'] = [len(k.split(' '))/4.0 for k in state['phrases']]
    #     state['phrases_first_encounters'] = [(float(k['first_encounter']) if float(
    #         k['first_encounter']) >= 0 else state['w_count'])/state['w_count'] for k in state['kc_vals']]
    #     state['phrase_embeddings'] = self.sent_transformer_model.encode(state['phrases'])

    # def candidate_weighting(self, state, df={}):
    #     phrases_set = []
    #     for i in range(len(state['phrases'])):
    #         avg_score, max_score, max_score_index, five_points = calculateScoreInterpolation(
    #             state['part_embeddings'], state['phrase_embeddings'][i])
    #         inp = [
    #             state['phrases_lengths'][i],
    #             state['phrases_counts'][i],
    #             state['phrases_first_encounters'][i],
    #             df[state['kc_lemmas'][i]] if state['kc_lemmas'][i] in df else 0.003
    #         ]+five_points
    #         # extract a batch from dataset
    #         x = torch.tensor(inp, device=self.device)
    #         self.kw_assessment_model.eval()
    #         # Forward pass: compute predicted y by passing x to the model.
    #         y_pred = self.kw_assessment_model(x)
    #         y_bin = (y_pred > 0.5).float()
    #         phrases_set.append({
    #             'phrase': state['phrases'][i],
    #             'features': inp,
    #             'score': y_pred.item(),
    #             'class': 'keyword' if y_bin == True else 'non-keyword'
    #         })
    #     phrases_set.sort(key=lambda x: x['score'], reverse=True)
    #     return phrases_set
