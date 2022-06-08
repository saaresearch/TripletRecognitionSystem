from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
from script import load_image
from script import load_class_names
from pdd.data_utils import load_config, prepare_datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder
from script import load_class_names

def main(opt):
    y_pred = []
    y_true = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config('config/script_percep_param.yaml')
    model = torch.jit.load(config['model'])
    if opt.split:
        _,test_ds = prepare_datasets('data/')

    else:
        TRANSFROMS_TEST = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

        test_ds = ImageFolder(opt.datatest, transform=TRANSFROMS_TEST)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=1,
                                            shuffle=False, num_workers=0)


    # iterate over test data
    for inputs, labels in tqdm(testloader):
        # inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = load_class_names(config['class_names'])

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    print(classification_report(y_true, y_pred))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    print('save fig')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=False, action="store_true", help='spit dataset to train and test')
    parser.add_argument('--datatest', type=str, default='', help='test data path')
    opt = parser.parse_args()
    main(opt)