# Few-Shot Semantic Segmentation
The full blogpost can be found here: [CLICK ME](docs\Pages\blogpost.html)

## File structure
Since there are a  great many files, the following diagram will show where important files are located in order to run this code yourself.\
.root\
├── docs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # contains all the files to run the networks\
│├─ Data\
││&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Annotaions &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Folder for the mask of each image\
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