@REM change nlp-model

@REM sed -i "s/START_ARR=.*/START_ARR=6/" 4_train_and_evaluate.py
@REM sed -i "s/END_ARR=.*/END_ARR=32/" 4_train_and_evaluate.py
@REM sed -i "s/H=.*/H=450/" 4_train_and_evaluate.py

@REM sed -i "s/NLP_MODEL=.*/NLP_MODEL=\"distilbert-multilingual-nli-stsb-quora-ranking\"/" 3_prepare_dataset_public_ds.py
@REM python 3_prepare_dataset_public_ds.py && python 4_train_and_evaluate.py > result/model_distilbert-multilingual-nli-stsb-quora-ranking.txt

@REM sed -i "s/NLP_MODEL=.*/NLP_MODEL=\"xlm-r-bert-base-nli-stsb-mean-tokens\"/" 3_prepare_dataset_public_ds.py
@REM python 3_prepare_dataset_public_ds.py && python 4_train_and_evaluate.py > result/model_xlm-r-bert-base-nli-stsb-mean-tokens.txt

@REM sed -i "s/NLP_MODEL=.*/NLP_MODEL=\"xlm-r-distilroberta-base-paraphrase-v1\"/" 3_prepare_dataset_public_ds.py
@REM python 3_prepare_dataset_public_ds.py && python 4_train_and_evaluate.py > result/model_xlm-r-distilroberta-base-paraphrase-v1.txt

@REM sed -i "s/NLP_MODEL=.*/NLP_MODEL=\"distiluse-base-multilingual-cased\"/" 3_prepare_dataset_public_ds.py
@REM python 3_prepare_dataset_public_ds.py && python 4_train_and_evaluate.py > result/model_distiluse-base-multilingual-cased.txt

@REM change features

sed -i "s/START_ARR=.*/START_ARR=7/" 4_train_and_evaluate.py
sed -i "s/END_ARR=.*/END_ARR=32/" 4_train_and_evaluate.py
python 4_train_and_evaluate.py > result/features_PL_TF_FA_SC.txt

sed -i "s/START_ARR=.*/START_ARR=6/" 4_train_and_evaluate.py
sed -i "s/END_ARR=.*/END_ARR=7/" 4_train_and_evaluate.py
python 4_train_and_evaluate.py > result/features_PL_TF_FA_DF.txt

sed -i "s/START_ARR=.*/START_ARR=6/" 4_train_and_evaluate.py
sed -i "s/END_ARR=.*/END_ARR=6/" 4_train_and_evaluate.py
python 4_train_and_evaluate.py > result/features_PL_TF_FA.txt

sed -i "s/START_ARR=.*/START_ARR=6/" 4_train_and_evaluate.py
sed -i "s/END_ARR=.*/END_ARR=32/" 4_train_and_evaluate.py
python 4_train_and_evaluate.py > result/features_PL_TF_FA_DF_SC.txt

@REM change neurons

@REM sed -i "s/H=.*/H=150/" 4_train_and_evaluate.py
@REM python 4_train_and_evaluate.py  result/4neurons_150.txt

@REM sed -i "s/H=.*/H=300/" 4_train_and_evaluate.py
@REM python 4_train_and_evaluate.py > result/8neurons_300.txt

@REM sed -i "s/H=.*/H=450/" 4_train_and_evaluate.py
@REM python 4_train_and_evaluate.py > result/8neurons_450.txt

@REM sed -i "s/H=.*/H=600/" 4_train_and_evaluate.py
@REM python 4_train_and_evaluate.py > result/8neurons_600.txt

@REM sed -i "s/H=.*/H=750/" 4_train_and_evaluate.py
@REM python 4_train_and_evaluate.py > result/8neurons_750.txt
