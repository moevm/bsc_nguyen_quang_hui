from zope.interface import implementer
import gc
from analizers.default_preprocessing_modules import create_default_encoder, create_default_nlp_module
from analizers import IncoherenceDetector, KeywordExtractor
import time

class ArticleAnalizingService(object):

    def __init__(self):
        encoder = create_default_encoder()
        language = 'ru'
        nlp_module = create_default_nlp_module(language=language)
        self.incoherence_detector = IncoherenceDetector(
            language=language, nlp_module=nlp_module)
        self.keyword_extractor = KeywordExtractor(
            language=language, nlp_module=nlp_module, encoder=encoder)
        print("service initiated")
        gc.collect()

    def evaluate_keyword(self, input, language, user_phrases=[]):
        document = self.keyword_extractor.load_document(
            input, language=language)
        return self.keyword_extractor.execute(document, is_evaluator=True, is_extractor=False, user_phrases=user_phrases)

    def detect_incoherence(self, input, language):
        document = self.incoherence_detector.load_document(
            input, language=language)
        return self.incoherence_detector.execute(document)

    def full_analyze(self, input, language, user_phrases=[], extractor_n = 10):
        start_time = time.time()
        document = self.keyword_extractor.load_document(
            input, language=language)
        load_time = time.time()-start_time
        start_time = time.time()
        self.keyword_extractor.execute(
            document, is_evaluator=True, is_extractor=True, user_phrases=user_phrases, n = extractor_n)
        keyword_time = time.time()-start_time
        start_time = time.time()
        self.incoherence_detector.execute(document)
        incoherence_time = time.time()-start_time
        result = {
            'processing_time':round(load_time+keyword_time+incoherence_time,3),
            'load_time':round(load_time,3),
            'keyword_time':round(keyword_time,3),
            'incoherence_time':round(incoherence_time,3),
            'coherence': list(map(lambda e: {
                'has_missing': e['has_missing'],
                'missing_sentences':e['missing_sentences'],
                'has_incoherent':e['has_incoherent'],
                'incoherent_sentences':e['incoherent_sentences'],
                'matrix_original':e['matrix_original']
            } if e['status']!='skipped' else {
            }, document['coherence'])),
            'user_phrases': list(map(lambda e: {
                'text':document['user_phrases'][e]['phrase'],
                'score':document['user_phrases'][e]['score']
            }, document['user_phrases'].keys())),
            'keyword_candidates': sorted(list(map(lambda e: {
                'text':document['keyword_candidates'][e]['phrase'],
                'score':document['keyword_candidates'][e]['score']
            },document['keyword_candidates'].keys())), key=lambda x: x['score'], reverse=True)[:extractor_n]
        }
        # print(result)
        return result

    # def example_function(self,name):
    #     self.count +=1
    #     return 'u:{0}{1}'.format(name, self.count)
