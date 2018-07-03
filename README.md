# mask-rcnn


This is a Pytorch 0.4 implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) that is in large parts based on Matterport's [Mask_RCNN](https://github.com/matterport/Mask_RCNN)\[1\] and (https://github.com/multimodallearning/pytorch-mask-rcnn)\[1\]. Matterport's repository is an implementation on Keras and TensorFlow.

The main improvements from \[2\] are:
- Pytorch 0.4
- supports batchsize > 1
- some bugs were fixed in the translation process

Currently, it works with Kaggle's 2018 Data Science Bowl dataset:

to train the network use:
python samples/nuclei.py train --dataset=path_to_dataset --model=coco

to detect use:
python samples/nuclei.py detect --dataset=path_to_dataset --model=path_to_trained_model

for installation instructions, please see \[2\]
