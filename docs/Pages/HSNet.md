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

To train HSNet on your own data will take time. PASCAL-5i (20,000 images) typically takes around two days on four 2080 Ti GPUs. Larger datasets such as COCO-20i (328,000 images) take a week on four Titan RTX GPUs. However, if the network has been trained before, on images that look similar to yours, no retraining is necessary. The testing samples must be within distribution of training for to use a pre-trained network. A PASCAL trained network will not work with non image data (such as an MRI scan).

A very important aspect of this training process is that all of the pre-trained networks I have used were always trained with an instance of the novel class in the query image. What this means is that in practice the network assumes that there must be an instance of the novel class in the image somewhere. When that is not the case, it does not return an empty mask but will find what it thinks the new class. For example, when trying to segment out people from an image taken of a street the network might segment out traffic bollards, signs, or even parked cars close to the pavement when there are no people in sight. 

### Tuning
It is possible to further improve the networks performance on a dataset through use of tuning. Tuning is essentially unfreezing the networks layers and training on new data. This can make it slightly more able to identify classes in new datasets. 

## Comparison to CNN
As the problem stemmed from an issue that Prorail had, it seemed fitting to use the detection of people as the comparison task. For this, I created a subset of 256 images from the validation set of "Cityscapes2" to test HSNet against a network trained specifically to segment, among other things, people from a different dataset. The conventional network I chose is  deeplabv3 resnet101. This is a pretrained network on the microsoft "Common Objects in Context" dataset achieving a mean intersection over union of 67.4%. I chose this network because it is easy to import from torchvision and therefore would be an obvious choice to most people who need a quick solution. When tested on the validation set, the conventional network achieved a mean intersection over union of XXX. 

On the other hand, HSNet, having never seen a person before, achieved a mean intersection over union of XXX. While this is very impressive, let's dive deeper into the results of the segmentation. In the figure below we see 

## Qualitative Analysis

<!-- 
Make a histogram that shows the distribution of the size of the masks
could perhaps compare the performance against the size of the person in the query
could look at what happens when the threshold is turned wayyy up, what is it most confident about

 -->