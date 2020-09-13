import torch.quantization
import torch.nn as nn
import torch
import os
import torch.nn.functional as F

from pdd.model import MLP
from pdd.model import PDDModel
from pdd.data_utils import load_config
from pdd.model import get_trained_model
from pdd.model import get_device
from collections import OrderedDict

 
def main():
   
    config = load_config('config/trace_parameters.yaml')
    embedding_model = PDDModel(1280, config['num_classes'], True)
    classifier_model = MLP(1280, config['num_classes'])
    device = get_device()
    model= get_trained_model(embedding_model, 'for_quant_PDD.pt',device)
    qmodel=torch.quantization.quantize_dynamic(model,dtype=torch.qint8)
    torch.save(qmodel.state_dict(), "qmodel.pt")
    print('Size (MB):', os.path.getsize("qmodel.pt")/1e6)
    print('Size (MB):', os.path.getsize("for_quant_PDD.pt")/1e6)
    qmodel = nn.Sequential(OrderedDict([
        ('embedding', qmodel),
        ('classifier', get_trained_model(classifier_model,
                                         'for_quant_clf.pt', device))]))
    torch.save(qmodel.state_dict(), "qmodel.pt")
    scripted_model = torch.jit.trace(qmodel, torch.rand(1, 3, 256, 256))
    scripted_model.save("qmodel.pt")



if __name__ == "__main__":

    main()