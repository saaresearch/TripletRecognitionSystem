from pdd.model import Perceptron_classifier
from pdd.model import PDDModel
from script import load_image
from pdd.data_utils import load_config
from pdd.model import get_trained_model
import torch.nn as nn
from collections import OrderedDict
import torch


def get_trace_model(embedding_model_path, classifier_model_path, save_path, device, num_classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = PDDModel(1280, num_classes, True)
    classifier_model = Perceptron_classifier(1280, num_classes)
    model = nn.Sequential(OrderedDict([
            ('embedding', get_trained_model(embedding_model, embedding_model_path, device)),
            ('classifier', get_trained_model(classifier_model, classifier_model_path, device)),
    ]))
    scripted_model = torch.jit.trace(model, torch.rand(1,3,256,256))
    scripted_model.save(save_path)


def main():
    config = load_config('config/trace_parameters.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_trace_model(config['embedding_model'], config['classifier_model'], config['save_model'], device, config['num_classes'])


if __name__ == "__main__":

    main()