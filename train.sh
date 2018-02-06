DATASET_DIR=../../storage/original/
TRAIN_DIR=./output/
CHECKPOINT_PATH=./data/resnet_v2_50.ckpt

python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=70 \
    --learning_rate=0.02 \
    --model_name=resnet_v2_50 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --train_image_size=180 \
    --checkpoint_exclude_scopes=resnet_v2_50/logits/
