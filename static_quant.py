import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
from pdd.quant_model import PDDModel
from pdd.quant_model import get_trained_model


def main():

    embedding_model = PDDModel(1280, 25, True)
    embedding_model = get_trained_model(
        embedding_model, 'triplet_model_param.pt', 'cpu')
    embedding_model.eval()

    embedding_model.fuse_model()
    embedding_model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(embedding_model, inplace=True)

    torch.quantization.convert(embedding_model, inplace=True)
    torch.save(embedding_model.state_dict(), 'quant_config1.pt')

    embedding_model = PDDModel(1280, 25, True)
    embedding_model = get_trained_model(
        embedding_model, 'static_quant.pt', 'cpu')
    embedding_model.eval()
    embedding_model.fuse_model()

    embedding_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(embedding_model, inplace=True)
    torch.quantization.convert(embedding_model, inplace=True)
    torch.save(embedding_model.state_dict(), 'quant_config2.pt')


main()
