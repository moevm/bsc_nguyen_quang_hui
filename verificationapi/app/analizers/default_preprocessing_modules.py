import stanza
from sentence_transformers import SentenceTransformer

def create_default_encoder():
    # loading SBERT model
    sent_transformer_model = SentenceTransformer(
        'xlm-r-bert-base-nli-stsb-mean-tokens')
    sent_transformer_model.max_seq_length = 200
    # 'distiluse-base-multilingual-cased' , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'distilbert-multilingual-nli-stsb-quora-ranking'
    return sent_transformer_model

def create_default_nlp_module(language='en'):
    return stanza.Pipeline(
        lang=language, processors='tokenize,pos,lemma')  # ,depparse,ner
