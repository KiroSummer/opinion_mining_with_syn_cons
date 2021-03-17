export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

CONFIG="config.json"
MODEL="model"

TRAIN_PATH="../data/aaai19srl.train0.conll.json"
#TRAIN_PATH="../data/aaai19srl.dev0.conll.json"
DEV_PATH="../data/aaai19srl.dev0.conll.json"
GOLD_PATH="../data/english/srl/conll05/conll05.devel.props.gold.txt"

CONS_PATH="../data/constituent_conll12/ontonote5.0.train.constituents.json"
DEP_TREES="/data2/qrxia/SRL-w-Heterogenous-Dep/data/english/dependency/ptb_from_baidu_from_n171/ptb.english.conll.train.txt.opentest.tag.projective"

SYS_CONS="../data/sentences/orl.2.0.all0.sentences.txt.constituent.txt"
SYS_DEP="../data/dependency_trees/orl.2.0.auto.dep.txt"

gpu_id=$1
CUDA_VISIBLE_DEVICES=$gpu_id python3 ../src/orl-4.1/train.py \
   --info="orl baseline bert" \
   --config=$CONFIG \
   --span="span" \
   --model=$MODEL \
   --train=$TRAIN_PATH \
   --dev=$DEV_PATH \
   --gold=$GOLD_PATH \
   --cons_trees=$CONS_PATH \
   --dep_trees=$DEP_TREES \
   --auto_cons_trees=$SYS_CONS \
   --auto_dep_trees=$SYS_DEP  \
   --gpu=$1
