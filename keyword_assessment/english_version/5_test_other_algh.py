import pke
import stanza
import json
from analizer import KeywordExtractor
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string


ps = PorterStemmer()
# nlp = stanza.Pipeline(
#     lang='en', processors='tokenize,lemma')  # ,depparse,ner
DATASET='nus'
DATASET_FILE='./dataset_'+DATASET+'.txt'

with open(DATASET_FILE, encoding='utf-8') as fp:
    datafiles = json.load(fp)

datafiles = [d for d in datafiles if d is not None]
print(len(datafiles))
START_FROM=150
datafiles = datafiles[START_FROM:]
count = 0



true_positive = 0
false_positive = 0
false_negative=0
true_negative=0
MODE=2
LIMIT=3
for i in range(len(datafiles)):
    keyphrases = datafiles[i]['keywords']
    text = "\n".join(datafiles[i]['fulltext'])
    
    if MODE == 1:
        # TEXTRANK
        extractor = pke.unsupervised.TextRank()
        extractor.load_document(input=text, language='en')
        extractor.candidate_weighting(window=2,top_percent=0.33, normalized=True)
    elif MODE == 2:
        # CURRENT_VERSION
        extractor = KeywordExtractor()
        extractor.load_document(input=text, language='en')
        extractor.candidate_selection()
        with open("./DF.txt", encoding='utf-8') as fp:
            df = json.load(fp)
        extractor.candidate_weighting(df=df)
    elif MODE == 3:
        # TFIDF
        extractor = pke.unsupervised.TfIdf()
        extractor.load_document(input=text, language='en')
        stoplist = stopwords.words('english')
        extractor.candidate_selection(n=3,stoplist=stoplist)
        df = pke.load_document_frequency_file(input_file='./df-weight.tsv.gz')
        extractor.candidate_weighting(df=df)
    elif MODE == 4:
        # KEA
        extractor = pke.supervised.Kea()
        extractor.load_document(input=text, language='en')
        stoplist = stopwords.words('english')
        df = pke.load_document_frequency_file(input_file='./df-weight.tsv.gz')
        extractor.candidate_selection(stoplist=stoplist)
        extractor.candidate_weighting(df=df)
    elif MODE == 5:
        # MULTIPARTITE
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text, language='en')
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(stoplist=stoplist)
        extractor.candidate_weighting(alpha=1.1,threshold=0.74, method='average')
    elif MODE == 6:
        # PositionRank
        pos = {'NOUN', 'PROPN', 'ADJ'}
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(input=text, language='en')
        extractor.candidate_selection(grammar=grammar,
                                    maximum_word_number=3)
        extractor.candidate_weighting(window=10,
                                    pos=pos)


    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    best = extractor.get_n_best(n=LIMIT)
    extracted_keyphrases = [word for word, score in best]

    all_phrases = [word for word, score in extractor.get_n_best(n=10000)]

    # extracted_keyphrases = [" ".join([word.lemma for word in nlp(phrase).sentences[0].words]) for phrase in extracted_keyphrases]
    # all_phrases = [" ".join([word.lemma for word in nlp(phrase).sentences[0].words]) for phrase in all_phrases]

    keyphrases = ["".join([ps.stem(word) for word in word_tokenize(phrase)]) for phrase in keyphrases]
    extracted_keyphrases = ["".join([ps.stem(word) for word in word_tokenize(phrase)]) for phrase in extracted_keyphrases]
    all_phrases = ["".join([ps.stem(word) for word in word_tokenize(phrase)]) for phrase in all_phrases]

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

precision = float(true_positive)/(true_positive+false_positive)
recall = float(true_positive)/(true_positive+false_negative)
f1 = 2*precision*recall/(precision+recall)
print(precision, "  ", recall, "  ", f1)
    
    
