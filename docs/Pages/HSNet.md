# HSNet
## Overview
HSNet stands for Hypercorrelation Squeeze Network and works using an encoder-decoder structure. First, the query and support images are fed through a backbone convolutional neural network which produces pairs of feature maps (one map for the support and one for the query). The feature maps from the support image are masked so that irrelevant activations are not used. 

Next, each feature map pair is analyzed to create a 4D correlation tensor. While this may sound complex, it is really just a large tensor that contains how similar each activation from the support feature is to each activation from the query image. This is repeated for each feature map pair and all the 4D correlation tensors are joined together into one massive encoded context. A high level diagram can be found below:

![4D Convolution Diagram](Images\4D Convolution Rough.png)

Finally, the encoded context is decoded back to two dimensions using simple 2D convolutions and upsampling. This gives a probability per pixel of that pixel being in the novel class. For testing, each pixel is assigned to its highest probability class, therefore giving a mask location on the query image.

## Training
<!-- ecentric movement is fine -->
HSNet is not a very big network with only some 2.5 million parameters to train. In part, this is due to the backbone network used being pre-trained and frozen. In effect, the only thing that is trained are the 4D convolutions used to analyze the similarity between the query and support images. 

To train HSNet, a dataset will typically contain several folds, each with unique classes. A training episode consists of giving the network a query and support image pair. The pair is fed forward through the network and it makes a prediction as to where the new class is in the query image. Then the difference between the prediction and the actual mask is calculated and fed backwards through the network. Through this, the network will learn what similarities are important between the support and query image, as well as how to convert the encoded context into a 2D mask. 

Training on a dataset the size of PASCAL-5i (20,000 images) typically takes around two days on four 2080 Ti GPUs. For larger datasets such as COCO-20i (328,000 images) it takes a week on four Titan RTX GPUs.

## Qualitative Analysis
A banger of a network

## Comparison to CNN
probably worse but what do you expect with like 5 support images