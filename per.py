import torch
from script import load_image
from pdd.model import PDD
from pdd.model import PDDModel
from pdd.model import Perceptron_classifier
from script import load_class_names
from pdd.data_utils import load_config 

def get_predict(img_name, feature_extractor, classifier, embedding_size, num_classes, class_names):
    
    pred_list = []
    pred_class = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emmbeding_model = PDDModel(embedding_size,num_classes,True)
    classifier_model = Perceptron_classifier(embedding_size,num_classes)
    model=PDD( emmbeding_model, classifier_model, embedding_size, num_classes, 'triplet.pt', 'classifier.pt',device)
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
    config = load_config('config/script_percep_param.yaml')
    pred = get_predict(config['img_path'],
                       config['feature_extractor'],
                       config['classifier'],
                       config['embedding_size'],
                       config['num_classes'],
                       config['class_names'])
    
    show_predict(pred,config['topn'])
    
   
    # path = 'script_test/health/health2.jpg'
    # path2 = '/home/artem/pdd/script_test/health_corn/health_1.jpg'
    # path3 = 'script_test/wheat_yellow_rust/yellow_rust1.jpg'
   
    
    
    
   
main()
