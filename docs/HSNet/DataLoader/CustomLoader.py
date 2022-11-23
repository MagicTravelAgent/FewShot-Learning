import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

class DatasetCustom(Dataset):
    def __init__(self, datapath, transform, shot, use_original_imgsize, experiment):
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.datapath = datapath

        # images and annotations in the same file for ease
        self.img_path = os.path.join(datapath, 'JPEGImages/')
        self.ann_path = os.path.join(datapath, 'Annotations/')
        self.transform = transform

        # setting up the lists of querys and supports
        self.querys = []
        self.supports = []
        self.setup_lists(experiment)

        if(len(self.supports) < shot):
          print("NOT ENOUGH SUPPORT IMAGES")


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # getting an item means you can enumerate through it
        # the way I want it to work is having a list with the names of query images
        # so the index can just loo through the index of the list of query names
        # I also want a list of support images (maybe it can sample from them also?)
        idx %= len(self.querys)  # for testing, as n_images < 1000
        query_name, support_names = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float())

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs
                 }

        return batch

    def extract_ignore_idx(self, mask):
        class_id = 0

        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        # print(query_name,"q name")
        # print(support_names, "s names")
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        query_name = self.querys[idx]

        support_names = []

        # while True:  # keep sampling support set if query == support
        #     support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        #     if query_name != support_name: support_names.append(support_name)
        #     if len(support_names) == self.shot: break

        for i in range(self.shot):
          support_name = np.random.choice(self.supports, 1, replace=False)[0]
          support_names.append(support_name)

        return query_name, support_names

    def setup_lists(self, experiment):
      """populates the list of query and support image names from the experiment"""

      def read(append, experiment):
        file_list = self.datapath + experiment
        with open(file_list + "_" + append + ".txt", 'r') as f:
          file_list = f.read().split('\n')[:-1]
        return file_list
      
      self.supports = read("support", experiment)
      self.querys = read("query", experiment)
