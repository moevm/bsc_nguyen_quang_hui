# # -*- coding: utf-8 -*-
# import stanza
# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, AutoConfig, AutoModel, BertModel
# import json
# import numpy as np
# from itertools import product
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from pathlib import Path
# import os

# from sentence_transformers import SentenceTransformer

# from analizers.default_preprocessing_modules import create_default_encoder, create_default_nlp_module
# from analizers import IncoherenceDetector, KeywordExtractor

# def print_size_of_model(model):
#     torch.save(model.state_dict(), "temp.p")
#     print('Size (MB):', os.path.getsize("temp.p")/1e6)
#     os.remove('temp.p')

# @profile
# def function():
#     # encoder = create_default_encoder()
#     language='ru'
#     nlp_module = create_default_nlp_module(language=language, use_gpu=False)
#     incoherence_detector = IncoherenceDetector(language=language, nlp_module=nlp_module)
#     keyword_extractor = KeywordExtractor(language=language, nlp_module=nlp_module)

# if __name__=='__main__':
#     function()