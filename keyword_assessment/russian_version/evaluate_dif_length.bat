set PYTHONIOENCODING=utf8

@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 3_prepare_dataset_public_ds.py
@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 4_train_and_evaluate.py

@REM python 3_prepare_dataset_public_ds.py > preparing_mixed.txt
@REM python 4_train_and_evaluate.py > training_mixed.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py

@REM sed -i "s/DATASET=.*/DATASET='mixed'/" analizer.py

@REM ----------------CHANNGE START ELEMENT IN 5_test_other_algh TO 600 MANUALLY------------------------------------------------------------------

@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mixed/ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mixed/ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mixed/ru_model_3.validate.txt

@REM @REM ----------------------------------------------------------------------------------

@REM sed -i "s/DATASET=.*/DATASET='mshort'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mshort/ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mshort/ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mshort/ru_model_3.validate.txt

@REM @REM ----------------------------------------------------------------------------------
@REM sed -i "s/DATASET=.*/DATASET='medium'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/medium/ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/medium/ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/medium/ru_model_3.validate.txt

@REM @REM ----------------------------------------------------------------------------------
@REM sed -i "s/DATASET=.*/DATASET='mlong'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mlong/ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mlong/ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mlong/ru_model_3.validate.txt




@REM -------------------------------------------- for 50-point curve
@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 3_prepare_dataset_public_ds.py
@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 4_train_and_evaluate.py

@REM python 3_prepare_dataset_public_ds.py > preparing_50_mixed.txt
@REM python 4_train_and_evaluate.py > training_50_mixed.txt

@REM sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py

@REM sed -i "s/DATASET=.*/DATASET='mixed'/" analizer.py

@REM @REM ----------------CHANNGE START ELEMENT IN 5_test_other_algh TO 600------------------------------------------------------------------

@REM sed -i "s/START_ELE=.*/START_ELE=600/" 5_test_other_algh.py
@REM sed -i "s/DATASET=.*/DATASET='mixed'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mixed/_50_ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mixed/_50_ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mixed/_50_ru_model_3.validate.txt

@REM @REM @REM -------------CHANNGE START ELEMENT IN 5_test_other_algh BACK TO 0---------------------------------------------------------------------

@REM sed -i "s/START_ELE=.*/START_ELE=0/" 5_test_other_algh.py
@REM sed -i "s/DATASET=.*/DATASET='mshort'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mshort/_50_ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mshort/_50_ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mshort/_50_ru_model_3.validate.txt

@REM @REM @REM ----------------------------------------------------------------------------------
@REM sed -i "s/DATASET=.*/DATASET='medium'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/medium/_50_ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/medium/_50_ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/medium/_50_ru_model_3.validate.txt

@REM @REM @REM ----------------------------------------------------------------------------------
@REM sed -i "s/DATASET=.*/DATASET='mlong'/" 5_test_other_algh.py

@REM sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mlong/_50_ru_model_5.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mlong/_50_ru_model_7.validate.txt

@REM sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > compare_length/mlong/_50_ru_model_3.validate.txt


@REM ------------------------STATISTIC
sed -i "s/DATASET=.*/DATASET='mixed'/" 3_dataset_statistic.py
python 3_dataset_statistic.py > compare_length/mixed/statistic_mixed.txt
sed -i "s/DATASET=.*/DATASET='mlong'/" 3_dataset_statistic.py
python 3_dataset_statistic.py > compare_length/mlong/statistic_mlong.txt
sed -i "s/DATASET=.*/DATASET='medium'/" 3_dataset_statistic.py
python 3_dataset_statistic.py > compare_length/medium/statistic_medium.txt
sed -i "s/DATASET=.*/DATASET='mshort'/" 3_dataset_statistic.py
python 3_dataset_statistic.py > compare_length/mshort/statistic_mshort.txt