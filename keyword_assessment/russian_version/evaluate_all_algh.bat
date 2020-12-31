set PYTHONIOENCODING=utf8

@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 3_prepare_dataset_public_ds.py
@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 4_train_and_evaluate.py
@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 5_test_other_algh.py
@REM sed -i "s/DATASET=.*/DATASET='mixed'/" analizer.py

@REM python 3_prepare_dataset_public_ds.py > preparing_mixed.txt
@REM python 4_train_and_evaluate.py > training_mixed.txt

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/textrank_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/ru_model_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/tfidf_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/textrank_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/kea_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/multipartite_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/positionrank_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/yake_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/textrank_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/ru_model_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/tfidf_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/kea_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/multipartite_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/positionrank_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/yake_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/textrank_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/ru_model_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/tfidf_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/kea_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/multipartite_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/positionrank_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/mixed/yake_3.validate.txt

@REM sed -i "s/DATASET=.*/DATASET='short'/" 3_prepare_dataset_public_ds.py
@REM sed -i "s/DATASET=.*/DATASET='short'/" 4_train_and_evaluate.py
@REM sed -i "s/DATASET=.*/DATASET='short'/" 5_test_other_algh.py
@REM sed -i "s/DATASET=.*/DATASET='short'/" analizer.py

@REM python 3_prepare_dataset_public_ds.py > preparing_short.txt
@REM python 4_train_and_evaluate.py > training_short.txt

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/textrank_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/ru_model_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/tfidf_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/kea_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/multipartite_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/positionrank_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/yake_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/textrank_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/ru_model_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/tfidf_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/kea_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/multipartite_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/positionrank_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/yake_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/textrank_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/ru_model_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/tfidf_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/kea_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/multipartite_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/positionrank_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/short/yake_3.validate.txt

sed -i "s/DATASET=.*/DATASET='long'/" 3_prepare_dataset_public_ds.py
sed -i "s/DATASET=.*/DATASET='long'/" 4_train_and_evaluate.py
sed -i "s/DATASET=.*/DATASET='long'/" 5_test_other_algh.py
sed -i "s/DATASET=.*/DATASET='long'/" analizer.py

@REM python 3_prepare_dataset_public_ds.py > preparing_long.txt
@REM python 4_train_and_evaluate.py > training_long.txt

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py

sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
python 5_test_other_algh.py > compare_algh/long/textrank_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/ru_model_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/tfidf_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/kea_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/multipartite_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/positionrank_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/yake_5.validate.txt

sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/textrank_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/ru_model_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/tfidf_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/kea_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/multipartite_7.validate.txt

sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
python 5_test_other_algh.py > compare_algh/long/positionrank_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/yake_7.validate.txt

sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py

sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
python 5_test_other_algh.py > compare_algh/long/textrank_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/ru_model_3.validate.txt

sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
python 5_test_other_algh.py > compare_algh/long/tfidf_3.validate.txt

sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
python 5_test_other_algh.py > compare_algh/long/kea_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/multipartite_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/positionrank_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_algh/long/yake_3.validate.txt

@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 3_dataset_statistic.py
@REM python 3_dataset_statistic.py > statistic_mixed.txt
@REM sed -i "s/DATASET=.*/DATASET='long'/" 3_dataset_statistic.py
@REM python 3_dataset_statistic.py > statistic_long.txt
@REM sed -i "s/DATASET=.*/DATASET='short'/" 3_dataset_statistic.py
@REM python 3_dataset_statistic.py > statistic_short.txt
