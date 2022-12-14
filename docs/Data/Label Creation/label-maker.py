from PIL import Image
import numpy as np
import glob
import json
from tqdm import tqdm

annotation_loc = "D:/AI/Capgemini/CityScapes/gtFine/val/frankfurt/"

# Get all the file names that i need to use
name_list = glob.glob(annotation_loc+'*.png')
# taking the colour images of the annotations
print(name_list)
ids = np.flatnonzero(np.core.defchararray.find(name_list,"color")!=-1)
name_list = np.take(name_list, ids)

# conversion dict
name_dict = {}
support = []  # not all images have people and they should be excluded from the support set

# open each file in turn
for index in tqdm(range(len(name_list))):

    image_name = name_list[index]
    # open the image to select the people
    im = Image.open(image_name)
    pix = np.array(im)[:, :, :3]

    for i in range(pix.shape[0]):  # for every pixel:
        for j in range(pix.shape[1]):
            comp = tuple(pix[i, j])
            if comp == (220, 20, 60):
                # turn the people white
                pix[i, j] = [255, 255, 255]

            else:
                # turn the background black
                pix[i, j] = [0, 0, 0]

    im = Image.fromarray(pix)
    im = im.convert("1")  # save the new black and white image
    # save the masks in a new location
    im.save(annotation_loc+"persons/"+str(index)+'.png')

    # identify the name of the image
    name = image_name.split('\\')[-1]

    # make a note of the name changes in a dictionary
    name_dict[name[:-4]] = index
    if(np.sum(pix) > 1):  # make a note of which images had people in them
        support.append(index)

# save the name changes and the valid support queries
with open('docs/Data/Label Creation/name_changes.txt', 'w') as convert_file:
    convert_file.write(json.dumps(name_dict))

with open('docs/Data/Label Creation/support_list.txt', 'w') as convert_file:
    convert_file.write('\n'.join(str(line) for line in support))
