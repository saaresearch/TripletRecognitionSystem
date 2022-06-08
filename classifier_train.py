from pyexpat import model
import torch
import torch.nn as nn
from torchbearer import Trial
from torch.optim import Adam
from torchbearer.callbacks import Best
from pdd.model import PDDModel
from pdd.data_utils import unzip_data
from pdd.data_utils import load_config
from pdd.model import get_trained_model
from pdd.data_utils import prepare_datasets, get_transform
from train import fix_random_seed
from pdd.train_test_split import split_on_train_and_test
from pdd.model import MLP
from pdd.trainer import forward_inputs_into_model
from collections import OrderedDict
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse



def train_classifier(model, optimizer, criterion, metrics,
                     train_features, train_labels, test_em, test_labels):
    checkpoint = Best(
        'classifier.pt',
        monitor='val_acc',
        mode='max',
        save_model_params_only=True)
    trial = Trial(
        model,
        callbacks=[checkpoint],
        optimizer=optimizer,
        criterion=criterion,
        metrics=['acc']
    )
    trial.with_train_data(
        torch.Tensor(train_features),
        torch.Tensor(train_labels).long())\
        .with_val_data(
        torch.Tensor(test_em),
        torch.Tensor(test_labels).long())\
        .for_steps(100)\
        .run(100)


def main(opt):
    device = 'cpu'
    config = load_config('config/train_parameters.yaml')
    config_script = load_config('config/script_parameters.yaml')
    fix_random_seed(config['random_seed'], config['cudnn_deterministic'])
    if opt.unzip:
        unzip_data(config['data_zip_path'], config['data_save_path'])
    if opt.split:
        # split_on_train_and_test(
        #     config['random_seed'],
        #     config['data_save_path'],
        #     config['test_size'])
        train_ds, test_ds = prepare_datasets(config['data_save_path'])
    else:
        train_ds, test_ds = get_transform(opt.datatrain, opt.datatest)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        pin_memory=True,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        pin_memory=True,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0)

    embedding_model = PDDModel(768, config['num_classes'], True)
    model_clf = MLP(768, config['num_classes'])
    embedding_model = get_trained_model(
        embedding_model, config_script['feature_extractor'], device)
    embedding_model.to(device)
    model_clf.to(device)
    test_em, test_labels = forward_inputs_into_model(test_loader, embedding_model,
                                                     device, config['batch_size'])
    train_em, train_labels = forward_inputs_into_model(train_loader, embedding_model,
                                                      device, config['batch_size'])

    optimizer = Adam(model_clf.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    train_classifier(
        model_clf,
        optimizer,
        criterion,
        ['acc'],
        train_em,
        train_labels,
        test_em,
        test_labels)
    full_model = nn.Sequential(OrderedDict([
        ('embedding', embedding_model),
        ('classifier', get_trained_model(model_clf,
                                         'classifier.pt', device))]))
    full_model = torch.jit.trace(full_model, torch.rand(1, 3, 256, 256))                                   
    traced_script_module = optimize_for_mobile(full_model)
    torch.jit.save(traced_script_module,'model.pt')
    traced_script_module._save_for_lite_interpreter("Mobilemodel.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=False, action="store_true", help='spit dataset to train and test')
    parser.add_argument('--unzip', default=False, action="store_true", help='unzip folder with dataset')
    parser.add_argument('--datatrain', type=str, default='', help='train data path')
    parser.add_argument('--datatest', type=str, default='', help='test data path')
    opt = parser.parse_args()

    main(opt)
