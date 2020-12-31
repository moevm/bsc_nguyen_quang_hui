sed -i "s/DATASET=.*/DATASET='nus'/" 3_prepare_dataset.py
sed -i "s/DATASET=.*/DATASET='nus'/" 4_train_and_evaluate.py
sed -i "s/DATASET=.*/DATASET='nus'/" 5_test_other_algh.py
sed -i "s/START_FROM=.*/START_FROM=150/" 5_test_other_algh.py

@REM python 3_prepare_dataset.py > result_nus/eng_model.prepare.txt
python 4_train_and_evaluate.py > result_nus/eng_model.trainresult.txt

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/textrank_5.validate.txt

sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
python 5_test_other_algh.py > result_nus/eng_model_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/tfidf_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/kea_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/multipartite_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/positionrank_5.validate.txt

sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/textrank_7.validate.txt

sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
python 5_test_other_algh.py > result_nus/eng_model_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/tfidf_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/kea_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/multipartite_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/positionrank_7.validate.txt

sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/textrank_3.validate.txt

sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
python 5_test_other_algh.py > result_nus/eng_model_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/tfidf_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/kea_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/multipartite_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_nus/positionrank_3.validate.txt


sed -i "s/DATASET=.*/DATASET='inspec'/" 3_prepare_dataset.py
sed -i "s/DATASET=.*/DATASET='inspec'/" 4_train_and_evaluate.py
sed -i "s/DATASET=.*/DATASET='inspec'/" 5_test_other_algh.py
sed -i "s/START_FROM=.*/START_FROM=1500/" 5_test_other_algh.py

@REM python 3_prepare_dataset.py > result_inspec/eng_model.prepare.txt
python 4_train_and_evaluate.py > result_inspec/eng_model.trainresult.txt

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
sed -i "s/LIMIT=.*/LIMIT=5/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/textrank_5.validate.txt

sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
python 5_test_other_algh.py > result_inspec/eng_model_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/tfidf_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/kea_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/multipartite_5.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/positionrank_5.validate.txt

sed -i "s/LIMIT=.*/LIMIT=7/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/textrank_7.validate.txt

sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
python 5_test_other_algh.py > result_inspec/eng_model_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/tfidf_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/kea_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/multipartite_7.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/positionrank_7.validate.txt

sed -i "s/LIMIT=.*/LIMIT=3/" 5_test_other_algh.py

@REM sed -i "s/MODE=.*/MODE=1/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/textrank_3.validate.txt

sed -i "s/MODE=.*/MODE=2/" 5_test_other_algh.py
python 5_test_other_algh.py > result_inspec/eng_model_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=3/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/tfidf_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=4/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/kea_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=5/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/multipartite_3.validate.txt

@REM sed -i "s/MODE=.*/MODE=6/" 5_test_other_algh.py
@REM python 5_test_other_algh.py > result_inspec/positionrank_3.validate.txt

