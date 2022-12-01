# HSNet
## Overview
HSNet stands for Hypercorrelation Squeeze Network and works using an encoder-decoder structure. As described in the overview, episodes of images are fed through a backbone convolutional neural network which produces pairs of feature maps (one map for the support and one for the query). The feature maps from the support image are masked so that irrelevant activations are not used. 

Next, each feature map pair is analyzed to create a 4D correlation tensor. The tensor in essence, is a measure of the similarity of the support features and query features to each other. This process is repeated for each feature map pair and all the 4D correlation tensors are joined together into an overall encoded context. A simplified diagram of feature pair comparison process can be found below:

![4D Convolution Diagram](Images\4D Convolution Rough.png)

Finally, the encoded context is decoded back to two dimensions using simple 2D convolutions and upsampling. This gives a probability per query image pixel of each pixel being in the novel class or not. At the testing stage, each pixel is assigned to its highest probability case (i.e. novel or not), thereby creating a mask location of the novel class on the query image.

## Training
<!-- ecentric movement is fine -->
With only 2.5 million parameters to train, HSNet is not large. This is due to the backbone of the network being pre-trained and frozen (its parameters are not updated during training). Training is restricted to the 4D convolutions and the context decoder only.

To train HSNet, a dataset will typically contain several folds, each fold with unique classes. In training, an episode is fed forward through the network and a prediction is made as to where the new class is in the query image.Then, the difference between the prediction and the correct mask is calculated and fed backwards through the network. The network subsequently  learns what similarities are relevant between the support and query features, as well as how to convert the encoded context into a 2D mask. For validation, the network can be tested against the unseen classes in the fold not used for training. 

To train HSNet on your own data will take time. PASCAL-5i (20,000 images) typically takes around two days on four 2080 Ti GPUs. Larger datasets such as COCO-20i (328,000 images) take a week on four Titan RTX GPUs. However, if the network has been trained before, on images that look similar to yours, no retraining is necessary. The testing samples must be within distribution of training for to use a re-trained network. A PASCAL trained network will not work with non image data (such as an MRI scan).

## Qualitative Analysis
Sticking to our initial project client of Prorail, lets use the network to identify people that could be on the tracks. 

## Comparison to CNN
probably worse but what do you expect with like 5 support images