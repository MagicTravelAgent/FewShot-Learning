# Few-Shot Semantic Segmentation
The full blogpost can be found here: [CLICK ME](https://magictravelagent.github.io/FewShot-Learning/docs/Pages/blogpost.html)

## File structure
Since there are a  great many files, the following diagram will show where important files are located in order to run this code yourself.\
.root\
├── docs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # contains all the files to run the networks\
│├─ Data\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Annotations &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Folder for the mask of each image\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── JPEGImages &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Folder for all support/query images\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Label Creation &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Scripts used to create support masks from the cityscapes 2 dataset\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test_query.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# text file containing all possible query images\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── test_support.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# text file containing all possible support images\
│├─ HSNet &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Files to run HSNet\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Common &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Where the evaluator and Visualizer are located \
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── __Model\res101_pas\res101_pas_fold3\best_model.pt__ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# here the [HSNet pt file](https://drive.google.com/drive/folders/1z4KgjgOu--k6YuIj3qWrGg264GRcMis2) (resnet101) should be placed \
│├─ MSANet &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Files to run MSANet\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── config &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# config files to set up MSANet architectures \
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── model &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# contains all the files for the MSANet model \
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── __ignore/__ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Here the [three pth files needed for MSANet](https://drive.google.com/file/d/1THnd0ZUX9k6PpMlO-W1Kjtn5K2a2iVP9/view?usp=share_link) should be placed\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── test.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Contains all the functions to run MSANet\
│└── output &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Folder for the output of the evaluations\
├── vis &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# output folder for the mask visualizations\
├── environment.yml &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# use this file to create an anaconda environment \
└── main.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Main file to start everything off

## Set Up
Before you can run the networks you must download a few different things that are too big to fit on the github repo. Firstly, the weights for HSNet must be downloaded from [here](https://drive.google.com/drive/folders/1z4KgjgOu--k6YuIj3qWrGg264GRcMis2) and placed into ``docs\HSNet\Model\res101_pas\res101_pas_fold3\best_model.pt``. Make sure to use the resnet101 version, otherwise you will have the dreaded "dimension mismatch" error due to some differences in network architecture. Next, you have to download [this](https://drive.google.com/file/d/1THnd0ZUX9k6PpMlO-W1Kjtn5K2a2iVP9/view?usp=share_link) zip file that contains three different ``pth`` files needed for MSANet. Unzip the contents, and place them into ``docs\MSANet\model\ignore\``.

Finally, you are ready to run the code, however first you must [create an anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using ``environment.yml``. This file contains the versions of each of the packages that I had installed when I tested the networks. 

## Experiment
Because the networks are so versatile, you are really able to use whatever images you want for support and query images, provided that they have masks. All of the available images should be placed into ``docs\Data\JPEGImages``, while the masks for each of these images should be placed into ``docs\Data\Annotations``. The support masks must have the same name as the image that they are masking. Colour images must be ``.jpg`` images, while support masks must be ``.png`` images with a bit-depth of 1. This forces the mask to be a binary, black and white, image.

Finally, you can choose which images can be selected to be in the support set, and which images you want to run segmentation on by adding the relevant file names into ``docs\Data\test_query.txt`` and ``docs\Data\test_support.txt``. The files must be only the file name with no extension and each file must be on a new line. Make sure that you have an empty line at the end of each file otherwise it will miss out on the last image.

## Running the network
Once everything is set up, you can run the network by first activating the environment that you created. Then you can run the ``main.py`` file by entering the command ``python main.py [--arguments]`` into the terminal. The arguments here change the settings of how you want to run your inference. The full list of arguments can be found in the ``main.py`` file but here is a quick overview of some important ones:
- ``--model``: This argument chooses which of the models you would like to run on your query.
- ``--test_size``: This argument limits the amount of query images you would like to test on. It will run through the test text file in order, and should the ``test_size`` be larger than the amount of query images, it will loop over the list, sampling new support images.
- ``--visualize``: This argument can be "True" or "False" depending on if you want to see what the network predicted the segmentation mask to be.
- ``--confidence_level``: This sets the threshold between 0 and 1 required for the CNN and HSNet (MSANet is a bit more complex), in order for a pixel to be classed as part of an instance of the novel class.