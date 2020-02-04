from pdd.model import PDDModel
import torch

def get_predict():
    model=PDDModel(1280,15,True)
    model.load_state_dict(torch.load(PATH))
    model.eval()

get_predict()