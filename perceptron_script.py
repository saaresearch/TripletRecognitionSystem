   
import torch
from script import load_image
from pdd.model import PDDModel
from pdd.model import Perceptron_classifier
from script import load_class_names
from pdd.data_utils import load_config
from pdd.model import Perceptron_classifier
from pdd.model import PDDModel
from script import load_image
from pdd.data_utils import load_config
from pdd.model import get_trained_model
import torch.nn as nn
from collections import OrderedDict
import torch


def get_predict(img_name, model, class_names, device):
    pred_list = []
    pred_class = []
    model.eval()
    img = load_image(img_name) 
    class_name = load_class_names(class_names)
    pred_list = model(img).sort(descending=True)[1].tolist()
    return pred_list[0]
    # for pred in range(len(pred_list[0])):
    #     pred_class.append(class_name[pred])
    # return pred_class
    # pred = model(img)
    # return class_name[torch.argmax(pred, dim=1)]

def string_show_predict(predictions,topn,class_names):
    class_name = load_class_names(class_names)
    top_pred = f'TOP {topn} Network prediction:\n'
    for top, pred in enumerate(predictions[:topn]):
        top_pred = top_pred + f'\t{top+1} {class_name[pred]}\n'
    return top_pred

def predforbot(image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config('config/script_percep_param.yaml')
    model = torch.jit.load(config['model'])
    pred = get_predict(image,
                       model,                                                   
                       config['class_names'],
                       device)  
    return (string_show_predict(pred,config['topn'],config['class_names']))

def show_predict(predictions,topn,class_names):
    class_name = load_class_names(class_names)
    print(f'TOP {topn} Network prediction:\n')
    for top, pred in enumerate(predictions[:topn]):
        print(f'\t{top+1} {class_name[pred]}')

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config('config/script_percep_param.yaml')
    model = torch.jit.load(config['model'])
    pred = get_predict(config['img_path'],
                       model,                                                   
                       config['class_names'],
                       device)               
    show_predict(pred,config['topn'],config['class_names'])


if __name__ == '__main__':
    main()
    # print(predforbot("image.jpg"))