from PIL import Image
import numpy as np
import glob
import json
from tqdm import tqdm

annotation_loc = "C:/Users/LPUDDIFO/OneDrive - Capgemini/Documents/Internship/Research/Datasets/Cityscapes/gtFine/val/frankfurt/"

# Get all the file names that i need to use
name_list = glob.glob(annotation_loc+'*.png')
# taking the colour images of the annotations
drop_mask = np.flatnonzero(np.core.defchararray.find(name_list,"color")!=-1)
name_list = np.take(name_list, drop_mask) 

# conversion dict
name_dict = {}
support = [] # not all images have people and they should be excluded from the support set

# open each file in turn
for index, image_name in tqdm(enumerate(name_list)):

    # open the image to select the people
    im = Image.open(image_name)
    pix = np.array(im)[:,:,:3]

    for i in range(pix.shape[0]): # for every pixel:
        for j in range(pix.shape[1]):
            comp = tuple(pix[i,j])
            if comp == (220, 20, 60):
                # turn the people white 
                pix[i,j] = [255, 255, 255]
                
            else:
                # turn the background black
                pix[i,j] = [0, 0, 0]

    im = Image.fromarray(pix) # save the new blakc and white image

    # save the masks in a new location
    im=im.convert("1")
    im.save(annotation_loc+"persons/"+str(index)+'.png')

    # identify the name of the image
    # print(image_name)
    name = image_name.split('\\')[-1]
    # print(name)

    # make a note of the name changes in a dictionary
    name_dict[name[:-4]] = index
    if(np.sum(pix) > 1):
        support.append(index)

# save the name changes and the valid support queries
with open('docs/Data/Label Creation/name_changes.txt', 'w') as convert_file:
     convert_file.write(json.dumps(name_dict))

with open('docs/Data/Label Creation/support_list.txt', 'w') as convert_file:
     convert_file.write('\n'.join(str(line) for line in support))