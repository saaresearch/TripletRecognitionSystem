import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from yaml import load
from yaml import FullLoader
from json import dump

TRANSFORMS_TRAIN = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

TRANSFROMS_TEST = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

def unzip_data(data_zip_path, data_path):
    os.system("unzip %s -d %s" % (data_zip_path, data_path))


def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)


def get_classname_file(classes_name):
    f = open('classes2.txt', 'w')
    s1 = '\n'.join(classes_name)
    f.write(s1)
    f.close()


def write_json_file(filename, data, indent=2):
    with open(filename, "w") as write_file:
        dump(data, write_file, indent=indent)


class AllCropsDataset(Dataset):
    def __init__(self, image_folder, subset='',
                 transform=None, target_transform=None):
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
        for i, directory in enumerate(os.listdir(image_folder)):
            self.crops.append(directory)
            # full path to the folder
            d_path = os.path.join(image_folder, directory, self.subset)
            # attribute name to set attribute
            attr_name = '%s_ds' % directory.lower()
            print("Load '%s' data" % attr_name)
            # set the attribute with the specified name
            setattr(self, attr_name, ImageFolder(d_path))
            # add the dataset to datasets list
            self.datasets.append(getattr(self, attr_name))
            # get dataset attribute
            ds = getattr(self, attr_name)
            # add attr targets to the global targets
            ds_targets = [x + len(self.classes) for x in ds.targets]
            self.targets.extend(ds_targets)
            # add particular classes to the global classes' list
            ds_classes = []
            for class_name in ds.classes:
                new_class = '__'.join([directory, class_name])
                self.class_to_idx[new_class] = len(
                    self.classes) + ds.class_to_idx[class_name]
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

def prepare_datasets(data_path):

    train_ds = AllCropsDataset(
        data_path,
        subset='train',
        transform=TRANSFORMS_TRAIN,
        target_transform=torch.tensor)

    test_ds = AllCropsDataset(
        data_path,
        subset='test',
        transform=TRANSFROMS_TEST,
        target_transform=torch.tensor)
    print('Train size:', len(train_ds))
    print('Test size:', len(test_ds))
    print('Number of samples in the dataset:', len(train_ds))
    print('Crops in the dataset:', train_ds.crops)
    print('Total number of classes in the dataset:', len(train_ds.classes))
    print('Classes with the corresponding targets:')
    print(train_ds.class_to_idx)

    return train_ds, test_ds

def get_transform(train_path, test_path):
    train_ds = ImageFolder(train_path, transform=TRANSFORMS_TRAIN)
    test_ds = ImageFolder(test_path, transform=TRANSFROMS_TEST)
    print('Train size:', len(train_ds))
    print('Test size:', len(test_ds))
    print('Number of samples in the dataset:', len(train_ds))
    # print('Crops in the dataset:', train_ds.crops)
    print('Total number of classes in the dataset:', len(train_ds.classes))
    print('Classes with the corresponding targets:')
    print(train_ds.class_to_idx)
    return train_ds, test_ds
