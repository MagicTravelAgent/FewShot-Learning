# FewShot-Learning

Currently, computer vision networks require hundreds of iterations over thousands of training samples to achieve human level performance on image segmentation tasks. Humans only need a few images to recognize any given object (we will call them classes) and are able to segment an image correctly after seeing only one training image with the new class. The field of “few-shot semantic segmentation” (FSS) attempts to close the gap between human and machine, by creating networks that need very few training images to segment a novel class in an image. 

In this blogpost I will cover three different state of the art (2022) networks, and explain how they work, train and how they can be applied to new situations. Finally, I will compare each of the networks to an existing convolutional neural network and evaluate how well each network performs.

Most FSS networks use "episodes" of support and query images. The network is given an image which contains an instance of the new class demarked by a filter (which we call a mask), as well as, a query image that it needs to segment the new class out of. Both images are then fed through what is known as a "backbone" network, which is a pre-trained convolutional neural network with the classification layers removed. Outputs from different layers are extracted to get found "features" from the backbone network. This ranges from simple shapes such as circles and lines to much higher level features such as textures. The found features from the support image are then compared the the found features from the query image and the network finds the areas in the query image that look similar to the new class features.

## [HSNet](docs\Pages\HSNet.html)
Hypercorrelation Squeeze Network

## [MSANet](docs\Pages\MSANet.html)
Multi-Similarity and Attention Guidance

## [LSeg](docs\Pages\LSeg.html)
Language-driven Semantic Segmentation

# Business use cases
make the moneyyy