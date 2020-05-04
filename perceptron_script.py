import torch
from script import load_image
from script import load_class_names
from pdd.data_utils import load_config


def get_predict(img_name, model, class_names, device):
    pred_list = []
    model.eval()
    img = load_image(img_name)
    pred_list = model(img).sort(descending=True)[1].tolist()
    return pred_list[0]


def show_predict(predictions, topn, class_names):
    class_name = load_class_names(class_names)
    print(f'TOP {topn} Network prediction:\n')
    for top, pred in enumerate(predictions[:topn]):
        print(f'\t{top+1} {class_name[pred]}')


def get_topn_pred(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config('config/script_percep_param.yaml')
    model = torch.jit.load(config['model'])
    pred = get_predict(img,
                       model,
                       config['class_names'],
                       device)
    show_predict(pred, config['topn'], config['class_names'])


if __name__ == '__main__':
    get_topn_pred('image.jpg')
