import numpy as np
import os

from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder


class AllCropsDataset(Dataset):
    def __init__(self, image_folder, subset='', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        # data subset (train, test)
        self.subset = subset
        # store each crop data
        self.datasets = []
        self.crops = []
        self.samples = []
        self.imgs = []
        self.classes = []
        self.targets = []
        self.class_to_idx = {}
        # iterate over all folders 
        # with all crops
        for i, d in enumerate(os.listdir(image_folder)):
            self.crops.append(d)
            # full path to the folder
            d_path = os.path.join(image_folder, d, self.subset)
            # attribute name to set attribute 
            attr_name = '%s_ds' % d.lower()
            print("Load '%s' data" % attr_name)
            # set the attribute with the specified name
            setattr(self, attr_name, ImageFolder(d_path))
            # add the dataset to datasets list
            self.datasets.append(getattr(self, attr_name))
            # get dataset attribute
            ds = getattr(self, attr_name)
            # add attr targets to the global targets
            ds_targets = [x+len(self.classes) for x in ds.targets]
            self.targets.extend(ds_targets)
            # add particular classes to the global classes' list
            ds_classes = []
            for c in ds.classes:
                new_class = '__'.join([d, c])
                self.class_to_idx[new_class] = len(self.classes) + ds.class_to_idx[c]
                ds_classes.append(new_class)
            self.classes.extend(ds_classes)
            # imgs attribute has form (file_path, target)
            ds_imgs, _ = zip(*ds.imgs)
            # images and samples are equal
            self.imgs.extend(list(zip(ds_imgs, ds_targets)))
            self.samples.extend(list(zip(ds_imgs, ds_targets)))
            
            
    def __len__(self):
        return len(self.samples)
      
      
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.datasets[0].loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target