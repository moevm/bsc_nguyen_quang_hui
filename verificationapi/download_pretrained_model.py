import stanza
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoConfig

_ = AutoConfig.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
_ = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

_ = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')
# 'distiluse-base-multilingual-cased' , 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens', 'distilbert-multilingual-nli-stsb-quora-ranking'

# stanza.download('en', processors='tokenize,pos,lemma')
stanza.download('ru', processors='tokenize,pos,lemma')
