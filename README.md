# FewShot-Learning

Currently in AI, computer vision networks require hundreds of iterations over thousands of training samples to achieve performance on images segmentation tasks that is similar to human level. However, people need only a few images to understand a given concept and are able to segment an image correctly after seeing only one training image with the new class. The field of “few-shot semantic segmentation” (FSS) attempts to close the gap between man and machine, by creating networks that need very few training images to segment a novel class in an image. 

In this blogpost I will cover three different state of the art (2022) networks, and explain how they work, train and how they can be applied to new situations. Finally, I will compare each of the networks to an existing convocational neural network and evaluate how well each network performs.

The basic idea that a lot of FSS network use is the use of "episodes" of support and query images. In essence, a network will be given an image with contains an instance of the new class demarked by a mask. The network also receives a query image that it needs to segment the new class out of. Usually, both images are fed through what is known as a "backbone" network, which is a pre trained convolutional neural network with the classification layers removed. Outputs from different layers are extracted to get "features" from the backbone network. This ranges from simple shapes such as circles and lines to much later layers that look for entire textures. The feature activations from the support image are then compared the the activations from the query image and the network finds the areas in the query image that "look similar" to the new class activations.

## [HSNet](docs\Pages\HSNet.html)
Hypercorrelation Squeeze Network

## [MSANet](docs\Pages\MSANet.html)
Multi-Similarity and Attention Guidance

## [LSeg](docs\Pages\LSeg.html)
Language-driven Semantic Segmentation