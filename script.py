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

def load_image(infilename) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float" )
    return data

            


def load_text_file(filename,places):
# open file and read the content in a list
    with open(filename, 'r') as filehandle:
        for line in filehandle:
             # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            places.append(currentPlace)

def get_predict(img_name):
    device = torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=PDDModel(1280,15,True)
    knn=KNeighborsClassifier(3,metric=cosine)
    model.load_state_dict(torch.load('triplet.pt', map_location='cpu'))
    # model=model.load('triplet.pt')
    img=load_image(img_name)
    resizeimg=resize(img,(256,256))
    resizeimg=resizeimg.reshape(1,3,256,256)
    inputs = torch.from_numpy(resizeimg).float()
    model.eval()
    print(type(inputs))
    transform=transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img=Image.open(img_name)   
    # img=resize(img,(256,256))                                 
    inp=transform(img)
    # print(inp.shape)
    inp=inp.reshape(1,3,256,256)
    # embedding=model(inputs).detach().cpu().numpy()
    embedding=model(inp).detach().cpu().numpy()
    knn= pickle.load(open('knn_model.sav', 'rb'))
    classes_name=[]
    load_text_file('classname.txt',classes_name)
    # print(classes_name)
    y_pred=knn.predict(embedding)
    # print(img.shape)
    print(resizeimg.shape)
    print(y_pred)
    print(classes_name[y_pred[0]])
    # print(type(y_pred.getattr())
    # print(classes_name[(int)y_pred])

get_predict("/home/artem/pdd/index.jpeg")
get_predict("xloros.jpg")
get_predict("gnil.jpg")





