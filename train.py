
import numpy as np
import torch
import time
import os
import random

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from torchvision import transforms
from torch import nn

from triplettorch import HardNegativeTripletMiner
from triplettorch import AllTripletMiner
from torch.utils.data import DataLoader
from triplettorch import TripletDataset

from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

from pdd.train_test_split import datadir_train_test_split
from pdd.data_utils import AllCropsDataset
from pdd.model import PDDModel
from pdd.trainer import TripletTrainer
from pdd.metrics import knn_acc
# from train_test_split import datadir_train_test_split
# from data_utils import AllCropsDataset
# from model import PDDModel
# from trainer import TripletTrainer

RANDOM_SEED=13
CUDN_DETERMENISTIC=True
NUM_CLASSES=15

# DATA_PATH = 'pdd'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ZIP_PATH = 'archive_full.zip'
DATA_PATH = 'data/'
TEST_SIZE = 0.2
N_SAMPLE= 8
BATCH_SIZE= 32
EPOCHS=5000



def fix_random_seed(seed, cudnn_determenistic=False):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=cudnn_determenistic

def unzip_data():
    os.system("unzip %s -d %s" % (DATA_ZIP_PATH, DATA_PATH)) 



def prepare_datasets():
    train_ds = AllCropsDataset(
        DATA_PATH, 
        subset='train',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4352, 0.5103, 0.2836], [0.2193, 0.2073, 0.2047])]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
          target_transform=torch.tensor)

    test_ds = AllCropsDataset(
        DATA_PATH, 
        subset='test',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        target_transform=torch.tensor)

    # print statistics
    print('Train size:', len(train_ds))
    print('Test size:', len(test_ds))
    print('Number of samples in the dataset:', len(train_ds))
    print('Crops in the dataset:', train_ds.crops)
    print('Total number of classes in the dataset:', len(train_ds.classes))
    print('Classes with the corresponding targets:')
    print(train_ds.class_to_idx)
    return train_ds, test_ds 
def split_on_train_and_test(random_seed):
    for crop in os.listdir(DATA_PATH):
        crop_path = os.path.join(DATA_PATH, crop)
        _ = datadir_train_test_split(crop_path, 
                                    test_size=TEST_SIZE, 
                                    random_state=random_seed)     




def main():

    fix_random_seed(RANDOM_SEED, CUDN_DETERMENISTIC)

    print("Extract data")
    unzip_data()

    print("Split on train and test")
    split_on_train_and_test(RANDOM_SEED)

    print("Create datasets")
    train_ds, test_ds = prepare_datasets()

    print("Create data loaders")
    train_set_d    = lambda index: train_ds[ index ][0].float( ).numpy( )
    test_set_d     = lambda index:  test_ds[ index ][0].float( ).numpy( )
    tri_train_set  = TripletDataset(torch.FloatTensor(train_ds.targets).numpy( ), train_set_d, len(train_ds), N_SAMPLE )
    tri_test_set   = TripletDataset(torch.FloatTensor(test_ds.targets),  test_set_d,  len(test_ds),1 )
    tri_train_load = DataLoader( tri_train_set,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2,
    pin_memory  = True
    )
    tri_test_load  = DataLoader( tri_test_set,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True
    )
  
    train_loader = torch.utils.data.DataLoader(train_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print("Create miner")
    miner=AllTripletMiner(.5).cuda()

    print("Build computational graph")
    model=PDDModel(1280,15,True)
    loss = torch.nn.NLLLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    print("Train model")
    fix_random_seed(RANDOM_SEED, CUDN_DETERMENISTIC)
    trainer = TripletTrainer(model=model,
                            optimizer=optimizer,
                            tri_train_load=tri_train_load,
                            epochs=5000,
                            tri_test_load=tri_test_load,
                            batch_size=BATCH_SIZE,
                            KNN_train_data_load=train_loader,
                            KNN_test_data_load=test_loader,
                            scheduler=scheduler,
                            nameofplotclasses=test_ds.classes,
                            num_classes=NUM_CLASSES)
                            
    trainer.train()  
    
# if __name__ == "__main__":

main()   