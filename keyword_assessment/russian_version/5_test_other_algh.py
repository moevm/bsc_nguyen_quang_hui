# coding=utf-8
import pke
import stanza
import json
from analizer import KeywordExtractor
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from spacy_stanza import StanzaLanguage
import re

snlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma')
spacy_pipelines = StanzaLanguage(snlp)

ps_ru = SnowballStemmer("russian") 
ps_eng = SnowballStemmer("english") 

DATASET='mlong'
DATASET_FILE="./cyberleninka_"+DATASET+".txt"

with open(DATASET_FILE, encoding='utf-8') as fp:
    datafiles = json.load(fp)

datafiles = [d for d in datafiles if d is not None]
print(len(datafiles))
START_ELE=0
datafiles = datafiles[START_ELE:]
count = 0

true_positive = 0
false_positive = 0
false_negative=0
true_negative=0
MODE=2
LIMIT=3
for i in range(len(datafiles)):
    try:
        keyphrases = datafiles[i]['keywords']
        text = "\n".join(datafiles[i]['fulltext'])
        
        if MODE == 1:
            # TEXTRANK
            extractor = pke.unsupervised.TextRank()
            extractor.load_document(input=text, language='ru', spacy_model=spacy_pipelines)
            extractor.candidate_weighting(window=2,top_percent=0.33, normalized=True)
        elif MODE == 2:
            # CURRENT_VERSION
            extractor = KeywordExtractor(snlp)
            extractor.load_document(input=text, language='ru')
            extractor.candidate_selection()
            with open("./DF.txt", encoding='utf-8') as fp:
                df = json.load(fp)
            extractor.candidate_weighting(df=df)
        elif MODE == 3:
            # TFIDF
            extractor = pke.unsupervised.TfIdf()
            extractor.load_document(input=text, language='ru', spacy_model=spacy_pipelines)
            stoplist = stopwords.words('russian')
            extractor.candidate_selection(n=3,stoplist=stoplist)
            df = pke.load_document_frequency_file(input_file='./df-weight.tsv.gz')
            extractor.candidate_weighting(df=df)
        elif MODE == 4:
            # KEA
            extractor = pke.supervised.Kea()
            extractor.load_document(input=text, language='ru', spacy_model=spacy_pipelines)
            stoplist = stopwords.words('russian')
            df = pke.load_document_frequency_file(input_file='./df-weight.tsv.gz')
            extractor.candidate_selection(stoplist=stoplist)
            extractor.candidate_weighting(df=df)
        elif MODE == 5:
            # MULTIPARTITE
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=text, language='ru', spacy_model=spacy_pipelines)
            pos = {'NOUN', 'PROPN', 'ADJ'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('russian')
            extractor.candidate_selection(stoplist=stoplist)
            extractor.candidate_weighting(alpha=1.1,threshold=0.74, method='average')
        elif MODE == 6:
            # PositionRank
            pos = {'NOUN', 'PROPN', 'ADJ'}
            grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
            extractor = pke.unsupervised.PositionRank()
            extractor.load_document(input=text, language='ru', spacy_model=spacy_pipelines)
            extractor.candidate_selection(grammar=grammar,
                                        maximum_word_number=3)
            extractor.candidate_weighting(window=10,
                                        pos=pos)
        elif MODE == 7:
            # YAKE
            # 1. create a YAKE extractor.
            extractor = pke.unsupervised.YAKE()

            # 2. load the content of the document.
            extractor.load_document(input=text, language='ru', spacy_model=spacy_pipelines)

            # 3. select {1-3}-grams not containing punctuation marks and not
            #    beginning/ending with a stopword as candidates.
            stoplist = stopwords.words('russian')
            extractor.candidate_selection(n=3, stoplist=stoplist)

            # 4. weight the candidates using YAKE weighting scheme, a window (in
            #    words) for computing left/right contexts can be specified.
            window = 2
            use_stems = False # use stems instead of words for weighting
            extractor.candidate_weighting(window=window,
                                        stoplist=stoplist,
                                        use_stems=use_stems)


        # N-best selection, keyphrases contains the 10 highest scored candidates as
        # (keyphrase, score) tuples
        best = extractor.get_n_best(n=LIMIT)
        extracted_keyphrases = [word for word, score in best]

        all_phrases = [word for word, score in extractor.get_n_best(n=10000)]

        # extracted_keyphrases = [" ".join([word.lemma for word in nlp(phrase).sentences[0].words]) for phrase in extracted_keyphrases]
        # all_phrases = [" ".join([word.lemma for word in nlp(phrase).sentences[0].words]) for phrase in all_phrases]

        keyphrases = ["".join([ps_eng.stem(ps_ru.stem(word)) for word in word_tokenize(phrase)]) for phrase in keyphrases]
        extracted_keyphrases = ["".join([ps_eng.stem(ps_ru.stem(word)) for word in word_tokenize(phrase)]) for phrase in extracted_keyphrases]
        all_phrases = ["".join([ps_eng.stem(ps_ru.stem(word)) for word in word_tokenize(phrase)]) for phrase in all_phrases]

        keyphrases = list(dict.fromkeys(keyphrases))
        extracted_keyphrases = list(dict.fromkeys(extracted_keyphrases))
        all_phrases = list(dict.fromkeys(all_phrases))
        
        for ek in extracted_keyphrases:
            if ek in keyphrases:
                true_positive+=1
            else:
                false_positive+=1
        
        for ph in all_phrases:
            if not ph in keyphrases or not ph in extracted_keyphrases:
                true_negative+=1
        
        for ph in keyphrases:
            if not ph in extracted_keyphrases:
                false_negative+=1
        
        print(keyphrases)
        print(best)
        print(true_positive, false_positive, true_negative, false_negative)
    except Exception as e:
        print(e)

precision = float(true_positive)/(true_positive+false_positive)
recall = float(true_positive)/(true_positive+false_negative)
f1 = 2*precision*recall/(precision+recall)
print(precision, "  ", recall, "  ", f1)