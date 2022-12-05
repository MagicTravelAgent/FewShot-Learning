import json
from PIL import Image


image_loc = "D:/AI/Capgemini/CityScapes/Images/frankfurt/"
image_dest = "D:/AI/Capgemini/Repo/FewShot-Learning/docs/Data/JPEGImages"

# Get all the file names that i need to change to jpeg - in addition to the new names (in the dict)
# reading the data from the file
with open('D:/AI/Capgemini/Repo/FewShot-Learning/docs/Data/Label Creation/name_changes.txt') as f:
    data = f.read()
names = json.loads(data)

for name in names.keys():
    cut_name = "_".join(name.split("_")[:-2]) # removing the "_gtFine_color" from the name so i can select the image
    png_img = Image.open(image_loc + cut_name + "_leftImg8bit.png") #open the image with PIL
    png_img.save(str(names[name]) + ".jpg") # save the image as a jpeg to be used byt the network
