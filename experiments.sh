#! /bin/bash

mkdir Results

# n experiments
#n=5
#for e in $(seq 1 $n)
#do

# a model per epoch
name="CNN_max_filter_l2_4$e"
./train.py --train_path="../DDICorpus/Train" --checkpoint_file=$name --position_dim=5 --l2_reg_lambda=3 --filter_sizes=2,4,6 --num_epochs=40 > Results/$name.txt
for m in `ls -tr ./runs/$name/checkpoints/*.meta`
do
./test.py --test_path="../DDICorpus/Test/DDITask" --checkpoint_file=${m:0:-5} >> Results/$name.txt
done

#rm -r runs/

#done

# Curva de aprendizaje con baseline
./train.py --train_path=$sentence_paths_train --train_entity_path=$entities_paths_train --train_relation_path=$relations_paths_train --dev_path=$sentence_paths_dev --dev_entity_path=$entities_paths_dev --dev_relation_path=$relations_paths_dev --checkpoint_file="baseline" > Results/baseline.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline/checkpoints/" >> Results/baseline.txt

: <<'END'

# Evaluación de distintos numero de filtros
num_filters=( 100 200 300 )
for i in ${num_filters[@]}
do
./train.py --train_path=$sentence_paths_train --train_entity_path=$entities_paths_train --train_relation_path=$relations_paths_train --dev_path=$sentence_paths_dev --dev_entity_path=$entities_paths_dev --dev_relation_path=$relations_paths_dev --num_filters $i --checkpoint_file="num_filters"$i > Results/num_filters$i.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/num_filters"$i"/checkpoints/" >> Results/num_filters$i.txt
done

# Evaluación de distintos tamaño de filtros
filter_sizes=( 2 3 4 2,3,4 )
for i in ${filter_sizes[@]}
do
./train.py --train_path=$sentence_paths_train --train_entity_path=$entities_paths_train --train_relation_path=$relations_paths_train --dev_path=$sentence_paths_dev --dev_entity_path=$entities_paths_dev --dev_relation_path=$relations_paths_dev --filter_sizes $i --checkpoint_file="filter_sizes"$i > Results/filter_sizes$i.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/filter_sizes"$i"/checkpoints/" >> Results/filter_sizes$i.txt
done

# Evaluación de distintas tamaños para el random embedding
embedding_dim=( 100 300 500 )
for i in ${embedding_dim[@]}
do
./train.py --train_path=$sentence_paths_train --train_entity_path=$entities_paths_train --train_relation_path=$relations_paths_train --dev_path=$sentence_paths_dev --dev_entity_path=$entities_paths_dev --dev_relation_path=$relations_paths_dev --embedding_dim $i --checkpoint_file="embedding_dim"$i > Results/embedding_dim$i.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/embedding_dim"$i"/checkpoints/" >> Results/embedding_dim$i.txt
done

# Evaluación de distintas tamaños para el part-of-speech embeddings con random embedding
pos_dim=( 0 5 10 )
for i in ${pos_dim[@]}
do
./train.py --train_path=$sentence_paths_train --train_entity_path=$entities_paths_train --train_relation_path=$relations_paths_train --dev_path=$sentence_paths_dev --dev_entity_path=$entities_paths_dev --dev_relation_path=$relations_paths_dev --pos_dim $i --checkpoint_file="pos_dim"$i > Results/pos_dim$i.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/pos_dim"$i"/checkpoints/" >> Results/pos_dim$i.txt
done

# Evaluación de distintas tamaños para el position embeddings con random embedding
position_dim=( 0 5 10 )
for i in ${position_dim[@]}
do
./train.py --train_path=$sentence_paths_train --train_entity_path=$entities_paths_train --train_relation_path=$relations_paths_train --dev_path=$sentence_paths_dev --dev_entity_path=$entities_paths_dev --dev_relation_path=$relations_paths_dev --position_dim $i --checkpoint_file="position_dim"$i > Results/position_dim$i.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_dim"$i"/checkpoints/" >> Results/position_dim$i.txt
done

END