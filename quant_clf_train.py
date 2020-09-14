import torch
import torch.nn as nn
from torchbearer import Trial
from torch.optim import Adam
from torchbearer.callbacks import Best
from pdd.quant_model import PDDModel
from pdd.data_utils import unzip_data
from pdd.data_utils import load_config
from pdd.model import get_trained_model
from train import prepare_datasets
from train import fix_random_seed
from train import split_on_train_and_test
from pdd.model import MLP
from pdd.trainer import forward_inputs_into_model
from collections import OrderedDict


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
        .for_steps(200)\
        .run(50)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config('config/train_parameters.yaml')
    config_script = load_config('config/script_parameters.yaml')
    fix_random_seed(config['random_seed'], config['cudnn_deterministic'])
    unzip_data(config['data_zip_path'], config['data_save_path'])
    split_on_train_and_test(
        config['random_seed'],
        config['data_save_path'],
        config['test_size'])
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

    model = PDDModel(1280, len(train_ds.classes), True)
    print(len(train_ds.classes))
    model = get_trained_model(model, 'triplet_model_param.pt', 'cpu')
    model.to('cpu')
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model.model.features[1].conv

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for image, target in train_loader:
            output = model(image)
            loss = criterion(output, target)

    torch.quantization.convert(model, inplace=True)

    embedding_model = model

    model_clf = MLP(1280, config['num_classes'])
    test_em, test_labels = forward_inputs_into_model(test_loader, embedding_model,
                                                     device, config['batch_size'])
    train_em, train_labels = forward_inputs_into_model(train_loader, embedding_model,
                                                       device, config['batch_size'])

    optimizer = Adam(model_clf.parameters(), lr=0.0001)
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
    qmodel = nn.Sequential(OrderedDict([
        ('embedding', embedding_model),
        ('classifier', get_trained_model(model_clf,
                                         'classifier.pt', 'cpu'))]))
    qmodel = torch.jit.trace(qmodel, torch.rand(1, 3, 256, 256))
    qmodel.save("qmodel.pt")


if __name__ == "__main__":

    main()
