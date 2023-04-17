# FSRE reproduction


This repository is the official implementation of Fine-grained Semantics-aware Representation Enhancement (https://arxiv.org/pdf/2108.08829.pdf). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

📋  To install requirements:

```setup
pip install -r requirements.txt
```


## Training

📋 To train the model(s) in the paper, run this command:

```train
VISIBLE_DEVICES=0 python -m  train_ddp --data_path /Datasets/monodepth_benchmark/kitti_raw_sync/

```


## Evaluation

📋 To evaluate my model on ImageNet, run:

```eval
python export_gt_depth.py --data_path /Datasets/monodepth_benchmark/kitti_raw_sync/  --split eigen

python evaluate_depth.py --load_weights_folder weights/ --data_path /Datasets/monodepth_benchmark/kitti_raw_sync/ 

```


## Trained Models

You can download the model trained on kitti_raw_sync for 25 epochs:

- [Trained model](https://drive.google.com/file/d/1UFRBeWo4pRTO-rTFnfopNsXTNuwInQNu/view?usp=share_link).


## Results

Our model achieves the following performance on :



