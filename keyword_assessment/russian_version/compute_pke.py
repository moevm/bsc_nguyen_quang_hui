import os
import sys
import csv
import math
import glob
import pickle
import gzip
import json
import bisect
import codecs
import logging

from itertools import combinations, product
from collections import defaultdict

from pke.base import LoadFile, get_stopwords, get_stemmer_func

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from string import punctuation
import stanza
from spacy_stanza import StanzaLanguage
import re

snlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma')
spacy_pipelines = StanzaLanguage(snlp)
"""Compute Document Frequency (DF) counts from a collection of documents.

N-grams up to 3-grams are extracted and converted to their n-stems forms.
Those containing a token that occurs in a stoplist are filtered out.
Output file is in compressed (gzip) tab-separated-values format (tsv.gz).
"""

def compute_document_frequency(input_dir,
                               output_file,
                               extension='xml',
                               language='en',
                               normalization="stemming",
                               stoplist=None,
                               delimiter='\t',
                               n=3,
                               max_length=10**6,
                               encoding=None):
    """Compute the n-gram document frequencies from a set of input documents. An
    extra row is added to the output file for specifying the number of
    documents from which the document frequencies were computed
    (--NB_DOC-- tab XXX). The output file is compressed using gzip.
    Args:
        input_dir (str): the input directory.
        output_file (str): the output file.
        extension (str): file extension for input documents, defaults to xml.
        language (str): language of the input documents (used for computing the
            n-stem or n-lemma forms), defaults to 'en' (english).
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
        stoplist (list): the stop words for filtering n-grams, default to None.
        delimiter (str): the delimiter between n-grams and document frequencies,
            defaults to tabulation (\t).
        n (int): the size of the n-grams, defaults to 3.
        encoding (str): encoding of files in input_dir, default to None.
    """

    # document frequency container
    frequencies = defaultdict(int)

    # initialize number of documents
    nb_documents = 0
    count=0
    # loop through the documents
    for input_file in glob.iglob(input_dir + os.sep + '*.' + extension):
        print(nb_documents)
        # if count>10:
        #     break
        #logging.info('reading file {}'.format(input_file))
        try:
            # initialize load file object
            doc = LoadFile()

            # read the input file
            doc.load_document(input=input_file,
                            language=language,
                            normalization=normalization,
                            max_length=max_length,
                            encoding=encoding, spacy_model=spacy_pipelines)

            # candidate selection
            doc.ngram_selection(n=n)

            # filter candidates containing punctuation marks
            doc.candidate_filtering(stoplist=stoplist)

            # loop through candidates
            for lexical_form in doc.candidates:
                frequencies[lexical_form] += 1

            nb_documents += 1

            if nb_documents % 1000 == 0:
                logging.info("{} docs, memory used: {} mb".format(nb_documents,
                                                            sys.getsizeof(
                                                                frequencies)
                                                            / 1024 / 1024 ))
        except:
            print("ERR")
    # create directories from path if not exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # dump the df container
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:

        # add the number of documents as special token
        first_line = '--NB_DOC--' + delimiter + str(nb_documents)
        f.write(first_line + '\n')

        for ngram in frequencies:
            line = ngram + delimiter + str(frequencies[ngram])
            f.write(line + '\n')

# stoplist for filtering n-grams
stoplist=list(punctuation)

# compute df counts and store as n-stem -> weight values
compute_document_frequency(input_dir='cyberleninka',
                           output_file='df-weight.tsv.gz',
                           extension='txt',           # input file extension
                           language='ru',                # language of files
                           normalization="stemming",    # use porter stemmer
                           stoplist=stoplist,
                           encoding='utf-8')