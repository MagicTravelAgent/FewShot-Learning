# HSNet
## Overview
HSNet stands for Hypercorrelation Squeeze Network and works using an encoder-decoder structure. First, the query and support images are fed through a backbone convocational neural network which produces pairs of feature maps (one map for the support and one for the query). The feature maps from the support image are masked so that irrelevant activations are not used. 

Next, each feature map pair is analyzed to create a 4D correlation tensor. While this may sound complex, it is really just a large tensor that contains how similar each activation from the support feature is to each activation from the query image. This is repeated for each feature map pair and all the 4D correlation tensors are joined together into one massive encoded context. A high level diagram can be found below:

![4D Convolution Diagram](Images\4D Convolution Rough.png)

Finally, the encoded context is decoded back to two dimensions using simple 2D convolutions and upsampling. This gives a probability per pixel of that pixel being in the novel class. For testing, each pixel is assigned to its highest probability class, therefore giving a mask location on the query image.

## Training
ecentric movement is fine

## Qualitative Analysis
A banger of a network

## Comparison to CNN
probably worse but what do you expect with like 5 support images