
import numpy as np
import torch
import os
import random
import argparse

from loguru import logger
from torch.utils.data import DataLoader


from pdd.triplettorch import AllTripletMiner
from pdd.triplettorch import TripletDataset
from pdd.train_test_split import split_on_train_and_test
from pdd.data_utils import prepare_datasets, get_transform
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




def main(opt):

    config = load_config('config/train_parameters.yaml')

    fix_random_seed(config['random_seed'], config['cudnn_deterministic'])

    if opt.unzip:
        print("Extract data")
        unzip_data(config['data_zip_path'], config['data_save_path'])
    if opt.split:
        print("Split on train and test")
        # split_on_train_and_test(
        #     config['random_seed'],
        #     config['data_save_path'],
        #     config['test_size'])

        print("Create datasets")
        train_ds, test_ds = prepare_datasets(config['data_save_path'])
    else:
        train_ds, test_ds = get_transform(opt.datatrain, opt.datatest)


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
                                num_workers=0,
                                pin_memory=False
                                )
    tri_test_load = DataLoader(tri_test_set,
                               batch_size=config['batch_size'],
                               shuffle=False,
                               num_workers=0,
                               pin_memory=False
                               )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        pin_memory=False,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        pin_memory=False,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0)
    print("Create miner")
    miner = AllTripletMiner(.5).cuda()

    print("Build computational graph")
    model = PDDModel(1280, len(train_ds.classes), True, 'mobilenet')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=False, action="store_true", help='spit dataset to train and test')
    parser.add_argument('--unzip', default=False, action="store_true", help='unzip folder with dataset')
    parser.add_argument('--datatrain', type=str, default='', help='train data path')
    parser.add_argument('--datatest', type=str, default='', help='test data path')
    opt = parser.parse_args()

    main(opt)
