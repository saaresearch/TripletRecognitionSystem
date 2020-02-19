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


def get_predict(img_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PDDModel(1280, 15, True)
    knn = KNeighborsClassifier(3, metric=cosine)
    model.load_state_dict(torch.load('triplet.pt', map_location=device))
    inp = load_image(img_name)
    model.eval()
    embedding = model(inp).detach().cpu().numpy()
    knn = pickle.load(open('knn_model.sav', 'rb'))

    classes_name = load_text_file('classname.txt')
    y_pred = knn.predict(embedding)
    distances, indices = knn.kneighbors(embedding[[0]], n_neighbors=3)
    print(distances, indices)
    print(indices.ravel().__dir__())
    print(indices.data)
    print(y_pred)
    print(classes_name[y_pred[0]])


def main():
    # path=input()

    # get_predict(path)
    get_predict("/home/artem/pdd/script_test/xloros/xloros.jpg")


if __name__ == '__main__':
    main()


# get_predict("/home/artem/pdd/index.jpeg")
# get_predict("/home/artem/pdd/script_test/xloros/xloros.jpg")
# get_predict("/home/artem/pdd/script_test/xloros/xloros2.jpg")
# get_predict("/home/artem/pdd/script_test/xloros/xloros3.jpg")

# print(" ")

# get_predict("/home/artem/pdd/script_test/health/health1.jpeg")
# get_predict("/home/artem/pdd/script_test/health/health2.jpg")
# get_predict("/home/artem/pdd/script_test/health/health3.jpg")

# print (" ")

# get_predict("/home/artem/pdd/script_test/gnil/gnil.jpg")
# get_predict("/home/artem/pdd/script_test/gnil/gnil1.jpg")
# get_predict("/home/artem/pdd/script_test/gnil/gnil3.jpg")

# print(" ")

# get_predict("/home/artem/pdd/script_test/apopleks/apop1.jpg")
# get_predict("/home/artem/pdd/script_test/apopleks/apop1.jpg")
# get_predict("/home/artem/pdd/script_test/apopleks/apop1.jpg")

# get_predict("/home/artem/pdd/script_test/health_corn/health_1.jpg")
# get_predict("/home/artem/pdd/script_test/health_corn/health_2.jpg")

# get_predict("/home/artem/pdd/script_test/wheat_yellow_rust/yellow_rust1.jpg")

# get_predict("/home/artem/pdd/script_test/health_weat/health.jpg")
