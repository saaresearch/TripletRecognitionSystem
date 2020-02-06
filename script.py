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

def load_image(infilename) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def get_predict(img_name):

    device = torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=PDDModel(1280,15,True)
    knn=KNeighborsClassifier(3,metric=cosine)
   
    # model.load_state_dict(torch.load('triplet.pt'), map_location=device)
    img=load_image(img_name)
    
    resizeimg=resize(img,(256,256))
    resizeimg=resizeimg.reshape(1,3,256,256)
    inputs = torch.from_numpy(resizeimg).float()
    model.eval()
    embedding=model(inputs).detach().cpu().numpy()
    knn= pickle.load(open('knn_model.sav', 'rb'))
    y_pred=knn.predict(embedding)
    print(img.shape)
    print(resizeimg.shape)
    print(y_pred

get_predict("/home/artem/pdd/index.jpeg")




