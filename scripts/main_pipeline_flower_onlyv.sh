#!/bin/bash

cd ..

# custom config
DATA="data/oxford_flowers"
TRAINER=PromptFL
PRETRAINED=True
VANDT=True
WARMUP_EPOCH=10
LR=0.001
# COOP=fp16
#DATASET=$1
CFG=$1  # config file
CTP=$2  # class token position (end or middle)
NCTX=$3  # number of context tokens
NUM_EPOCH=$4
IID=$5
CSC=$6 # class-specific context (False or True)
USEALL=$7
SHOTS=$8
ONLY_VISUAL=True
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
#for DATASET in caltech101
for DATASET in oxford_flowers
do
  for SHOTS in ${SHOTS}
  do
    for REPEATRATE in 0.0
    do
      #for USERS in 64
      for USERS in 10
      do
        for EPOCH in ${NUM_EPOCH}
        do
          for ROUND in 10
          do
            for SEED in 1 2 3
            do
              DIR=output/${DATASET}_onlyv/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/pretrain_${PRETRAINED}/iid_${IID}_repeatrate_${REPEATRATE}/${USERS}_users/lr_${LR}/${EPOCH}epoch_${ROUND}round/seed${SEED}
              if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
              else
                python federated_main.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                TRAINER.PROMPTFL.N_CTX ${NCTX} \
                TRAINER.PROMPTFL.CSC ${CSC} \
                TRAINER.PROMPTFL.CLASS_TOKEN_POSITION ${CTP} \
                TRAINER.PROMPTFL.ONLY_VISUAL ${ONLY_VISUAL}\
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.USERS ${USERS} \
                DATASET.IID ${IID} \
                DATASET.REPEATRATE ${REPEATRATE} \
                OPTIM.MAX_EPOCH ${EPOCH} \
                OPTIM.ROUND ${ROUND}\
                OPTIM.LR ${LR}\
                OPTIM.WARMUP_EPOCH ${WARMUP_EPOCH}\
                MODEL.BACKBONE.PRETRAINED ${PRETRAINED}\
                DATASET.USEALL ${USEALL}\
                TRAINER.PROMPTFL.VANDT ${VANDT}
              fi
            done
          done
        done
      done
    done
  done
done