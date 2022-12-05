from PIL import Image
import numpy as np

annotation_location = "D:/AI/Capgemini/Repo/FewShot-Learning/docs/Data/Annotations/"
anno_sizes = []

# get all the support images
with open("D:/AI/Capgemini/Repo/FewShot-Learning/docs/Data/Label Creation/support_list.txt") as f:
    lines = f.read().splitlines()

# for each image count how many pixels it has and store it in a tuple with the file name
for item in lines:
    item_i = Image.open(annotation_location + item + ".png")
    item_i = np.asarray(item_i)
    anno_sizes.append((item, np.sum(item_i)))

# sort the list by amount of pixels in the support image
anno_sizes.sort(key = lambda x: x[1], reverse=True)

# get the names in the correct order
anno_names_sorted = np.array(anno_sizes)
anno_names_sorted = anno_names_sorted[:,0]

# print the sorted file:
with open("D:/AI/Capgemini/Repo/FewShot-Learning/docs/Data/Label Creation/support_list_sorted.txt", 'w') as fp:
    fp.write('\n'.join(anno_names_sorted))