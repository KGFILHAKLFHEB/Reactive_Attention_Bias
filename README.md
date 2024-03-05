# Reactive Attention Bias
Reference code for ECCV paper

## Requirements
```
pip3 install -r requirements.txt
```
The code was tested with Python 3.8.10

## How to run?
Pre-training is done with the following commands
```
$ python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model $MODEL --mixup 0.8 --cutmix 1.0 --batch-size 128 --epochs 300 --num_workers 40 --data-path $PATH --data-set CUB --output_dir $SAVE_PATH
```
Put the name of the model you want to train in $MODEL, the path to the dataset in $PATH, and the location where the checkpoints are saved in $SAVE_PATH.
The corresponding model names are shown below, respectively.
```
  ViT-S         :deit_small_patch16_224_12
  ViT-S + IAB   :deit_small_patch16_224_12_IAB
  ViT-S + RAB   :deit_small_patch16_224_12_RAB
  ViT-S + LRAB  :deit_small_patch16_224_12_LRAB
```

Fine-tuning is done with the following commands
```
$ python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model $MODEL --mixup 0.8 --cutmix 1.0 --batch-size 128 --epochs 100 --num_workers 40 --data-path $PATH --data-set CUB --finetune $SAVED_PATH --output_dir $SAVE_PATH
```
In '$SAVED_PATH, please put the path to the checkpoint saved in the pre-training


Tuning of ABs using human knowledge is done with the following commands
```
$ python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main_ht.py --model $MODEL --mixup 0.0 --cutmix 0.0 --batch-size 128 --epochs 100 --num_workers 40 --data-path $PATH --data-set CUBGHA --human --finetune $SAVED_PATH --output_dir $SAVE_PATH
```
In $SAVED_PATH, please put the path to the checkpoint saved in the fine-tuning with CUB-200-2010 or CUB-200-2011.
