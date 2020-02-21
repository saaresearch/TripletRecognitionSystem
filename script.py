from pdd.model import PDDModel
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
from skimage.transform import resize
import torch
from scipy.spatial.distance import cosine
import pickle
from torchvision import transforms
from torchvision import transforms, datasets
from pdd.data_utils import load_config
import json


def load_image(infilename):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(infilename)
    img = transform(img)
    img = img.reshape(1, 3, 256, 256)
    return img


def load_text_file(filename):
    classes_name = []
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            classes_name.append(currentPlace)
    return classes_name

def show_predict(filename):
    with open(filename) as f:
        items = json.load(f)
    print('\033[1m' + "Network prediction:")
    item_prediction = items['prediction']
    print('\033[0m' + "Class label:", item_prediction['label'])
    print("Class name:", item_prediction['class_name'])
    item_knn = items['knn parameters']
    print('\033[1m' + "KNN parameters:")
    print('\033[0m' + "Count neighbors:", item_knn['n_neighbors'])
    print('\033[0m' + "Metrics:", item_knn['metrics'])
    item_topn = items['topn']
    print('\033[1m' + "TOP 5 predictions :")
    for predict in item_topn:
        print('\033[0m' + "label:", str(predict['label']),
              "Image index:", str(predict['index']),
              "Distance:", str(predict['distance']), sep="  ")


def create_json_file(label, class_name, knn_model, distances, indices):

    data = {
        "prediction": {
            "label": str(label),
            "class_name": str(class_name[label])


        },
        "knn parameters":
        {
            "n_neighbors": str(knn_model.n_neighbors),
            "metrics": str(knn_model.metric.__name__)
        },
        "topn": [
            {"label": str(knn_model._y[indices[0][0]]), "index":str(
                indices[0][0]), "distance": "%.4f" % (distances[0][0])},
            {"label": str(knn_model._y[indices[0][1]]), "index":str(
                indices[0][1]), "distance": "%.4f" % (distances[0][1])},
            {"label": str(knn_model._y[indices[0][2]]), "index":str(
                indices[0][2]), "distance": "%.4f" % (distances[0][2])},
            {"label": str(knn_model._y[indices[0][3]]), "index":str(
                indices[0][3]), "distance": "%.4f" % (distances[0][3])},
            {"label": str(knn_model._y[indices[0][4]]), "index":str(
                indices[0][4]), "distance": "%.4f" % (distances[0][4])}

        ]

    }
    with open("data_file.json", "w") as write_file:
        json.dump(data, write_file, indent=2)


def get_predict(img_name, triplet_model_weight, knn_model_weight, class_names):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PDDModel(1280, 15, True)
    knn = KNeighborsClassifier(3, metric=cosine)
    model.load_state_dict(
        torch.load(
            triplet_model_weight,
            map_location=device))
    inputs = load_image(img_name)
    model.eval()
    embedding = model(inputs).detach().cpu().numpy()
    knn = pickle.load(open(knn_model_weight, 'rb'))

    classes_name = load_text_file(class_names)
    y_pred = knn.predict(embedding)
    distances, indices = knn.kneighbors(embedding[[0]], n_neighbors=10)
    create_json_file(y_pred[0], classes_name, knn, distances, indices)
    


def main():
    # path=input()
    # get_predict(path)
    config = load_config('config/script_parametrs.yaml')
    get_predict(
        config['img_path'],
        config['triplet_model_weight'],
        config['knn_model_weight'],
        config['class_name'])
    show_predict('data_file.json')

if __name__ == '__main__':
    main()
