# bsc_nguyen_quang_hui

## Method

+ Data collection
Collected several articles in the field computer and information science from cyberleninka (2000 articles in total)
Removed too short paragraphs, special symbol, literature reference, separate the keyword list from the text
+ Data preprocessing
The article's paragraphs and keywords are passed through a processor for tokenization, lemmarization, part of speech analization; passed through a multi language pretrained sentence embedder. In the progress, non-keyword noun phrases are also extracted and processed together with keywords and marked negative output (keywords are positive marked). Finally it is split into training set and test set (75-25). The training set is rebalanced (due to the number of non-keyword noun phrases vastly outnumbered the number of keywords, this will be classified an imbalanced classification problem, undersample or in this case, oversample is needed)
Input included 29 features: phrase's length (in words) (PL), number of occurrences (TF), position of first appearance in the text (FA), number of occurences in corpus (DF), 25-point interpolation of cosine similarity scores between phrase and paragraphs (CS)
+ Network training
Built a network with pytorch output a single number as confident score (answer the question: is this a good keyword?). Trained and evaluated with the dataset collected. 

# Result

With different combination of features
| Features | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| PL+TF+FA          | 0.1475 | 0.8101 | 0.2495 | 0.9371 |
| PL+TF+FA+DF       | 0.1336 | **0.8258** | 0.2301 | 0.9289 |
| PL+TF+FA+CS       | 0.3439 | 0.6537 | 0.4507 | 0.9795 |
| PL+TF+FA+DF+CS    | **0.3608** | 0.6526 | **0.4647** | **0.9805** |

From Table 1 we can see that the result with CS is higher than without it. 
Take a look at the result for articles, processed with and without DF: ("sample_result_450x3/" and "sample_result_450x3_tdidf/")
The results processed with DF parameter seems better, without short/common words.

With different number of layers and neurons in the network using all features
| Neuron per layer | Hidden layers | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 250 | 3 | 0.20 | 0.73 | * | 0.9* |
| 350 | 3 | 0.25 | 0.70 | * | 0.9* |
| 450 | 3 | 0.36 | 0.65 | * | 0.9* |
| 450 | 5 | 0.46 | 0.60 | * | 0.9* |
| 800 | 3 | 0.51 | 0.60 | * | 0.9* |
| 800 | 4 | 0.59 | 0.54 | * | 0.9* |
| 800 | 5 | 0.59 | 0.52 | * | 0.9* |

With increasing number of neurons per layer and number of layers, the recall value decrease but the precision increase. Meaning the criteria is more strict. Considering the Recall should not be too low, the result at 450 neurons per layers, 3 layers was chosen.
