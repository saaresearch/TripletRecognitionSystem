
import numpy as np
import torch
import os
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from triplettorch import AllTripletMiner
from triplettorch import TripletDataset

from pdd.train_test_split import datadir_train_test_split
from pdd.data_utils import AllCropsDataset
from pdd.model import PDDModel
from pdd.trainer import TripletTrainer
from pdd.data_utils import unzip_data
from pdd.data_utils import load_config
from pdd.model import get_device
from pdd.data_utils import get_classname_file

DEVICE = get_device()


def fix_random_seed(seed, cudnn_determenistic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_determenistic


def prepare_datasets(data_path):
    train_ds = AllCropsDataset(
        data_path,
        subset='train',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
        target_transform=torch.tensor)

    test_ds = AllCropsDataset(
        data_path,
        subset='test',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
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


def split_on_train_and_test(random_seed, data_path, test_size):
    for crop in os.listdir(data_path):
        crop_path = os.path.join(data_path, crop)
        _ = datadir_train_test_split(crop_path,
                                     test_size=test_size,
                                     random_state=random_seed)


def main():

    config = load_config('config/train_parameters.yaml')

    fix_random_seed(config['random_seed'], config['cudnn_deterministic'])

    print("Extract data")
    unzip_data(config['data_zip_path'], config['data_save_path'])

    print("Split on train and test")
    split_on_train_and_test(
        config['random_seed'],
        config['data_save_path'],
        config['test_size'])

    print("Create datasets")
    train_ds, test_ds = prepare_datasets(config['data_save_path'])
    print("Create data loaders")
    def train_set_d(index): return train_ds[index][0].float().numpy()
    def test_set_d(index): return test_ds[index][0].float().numpy()
    tri_train_set = TripletDataset(
        torch.FloatTensor(
            train_ds.targets).numpy(),
        train_set_d,
        len(train_ds),
        config['n_sample'])
    tri_test_set = TripletDataset(
        torch.FloatTensor(
            test_ds.targets),
        test_set_d,
        len(test_ds),
        1)
    tri_train_load = DataLoader(tri_train_set,
                                batch_size=config['batch_size'],
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True
                                )
    tri_test_load = DataLoader(tri_test_set,
                               batch_size=config['batch_size'],
                               shuffle=False,
                               num_workers=2,
                               pin_memory=True
                               )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        pin_memory=True,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        pin_memory=True,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2)
    print("Create miner")
    miner = AllTripletMiner(.5).cuda()

    print("Build computational graph")
    model = PDDModel(1280, len(train_ds.classes), True)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['step_size'], gamma=config['gamma'])
    print("Train model")
    fix_random_seed(config['random_seed'], config['cudnn_deterministic'])
    loss_history = []
    trainer = TripletTrainer(
        model=model,
        optimizer=optimizer,
        train_triplet_loader=tri_train_load,
        epochs=config['epochs'],
        test_triplet_loader=tri_test_load,
        batch_size=config['batch_size'],
        knn_train_loader=train_loader,
        knn_test_loader=test_loader,
        scheduler=scheduler,
        plot_classes_name=test_ds.classes,
        num_classes=config['num_classes'],
        miner=miner,
        loss_history=loss_history,
        safe_plot_img_path=config['plot_embeddings_img'],
        model_save_path=config['model_save_path'],
        optim_save_path=config['optim_save_path'],
        knn_metric=config['knn_metric']
    )
    get_classname_file(train_ds.classes)
    trainer.train()
    


if __name__ == "__main__":

    main()
