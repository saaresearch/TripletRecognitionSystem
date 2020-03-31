import torch
from script import load_image
from pdd.model import PDDModel
from pdd.model import Perceptron_classifier
from script import load_class_names
from pdd.data_utils import load_config 


def get_predict(img_name, model, class_names, device):
    pred_list = []
    pred_class = []
    model.eval()
    img = load_image(img_name) 
    class_name = load_class_names(class_names)
    pred_list = model(img).sort()[1].tolist()
    for pred in range(len(pred_list[0])):
        pred_class.append(class_name[pred])
    return pred_class  
    

def show_predict(predictions,topn):
    
    print(f'TOP {topn} Network prediction:\n')
    for top, pred in enumerate(predictions[:topn]):
        print(f'\t{top+1} {pred}')

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config('config/script_percep_param.yaml')
    model = torch.jit.load(config['model'])
    pred = get_predict(config['img_path'],
                       model,                                                   
                       config['class_names'],
                       device)  
    show_predict(pred,config['topn'])


if __name__ == '__main__':
    main()
