import torch
import torch.nn as nn
from pdd.model import PDDModel
from pdd.data_utils import unzip_data
from pdd.data_utils import load_config
from pdd.trainer import save_model
from pdd.model import get_trained_model
from train import prepare_datasets
from train import fix_random_seed
from train import split_on_train_and_test
from pdd.model import Perceptron_classifier
from torchbearer import Trial
from torch.optim import Adam
from pdd.trainer import forward_inputs_into_model


def train_classifier(model, optimizer, criterion, metrics, train_em, train_labels, test_em, test_labels):
    # optimizer = Adam(model.parameters())
    trial = Trial(model, optimizer=optimizer, criterion=criterion, metrics=['acc'])
    trial.with_train_data(torch.Tensor(train_em), torch.Tensor(train_labels).long()).with_val_data(torch.Tensor(test_em), torch.Tensor(test_labels).long()).for_steps(100).run(10)
    save_model(model, optimizer, 'classifier.pt', 'classifieroptim.pt')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config('config/train_parameters.yaml')
    config_script = load_config('config/script_parameters.yaml')
    fix_random_seed(config['random_seed'], config['cudnn_deterministic'])
    split_on_train_and_test(
        config['random_seed'],
        config['data_save_path'],
        config['test_size'])
    unzip_data(config['data_zip_path'], config['data_save_path'])
    train_ds, test_ds = prepare_datasets(config['data_save_path'])

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

    modelpdd = PDDModel(1280, config['num_classes'], True)
    modelclassifier = Perceptron_classifier(1280, config['num_classes'])
    modelpdd = get_trained_model(modelpdd, config_script['feature_extractor'],device)
    test_em, test_labels = forward_inputs_into_model(test_loader, modelpdd,
                                                             device, config['batch_size'])
    train_em, train_labels = forward_inputs_into_model(train_loader, modelpdd,
                                                               device, config['batch_size'])

    optimizer = Adam(modelclassifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_classifier(modelclassifier, optimizer, criterion, ['acc'], train_em, train_labels, test_em, test_labels)


if __name__ == "__main__":

    main()
