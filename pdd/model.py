import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, emmbeding_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def save_model(model, optimizer, model_save_path, optim_save_path):
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict, optim_save_path)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class PDDModel(nn.Module):

    """ 
    This model based on architecture MobileNetV2
    """
    def __init__(self, embedding_size, num_classes, pretrained=False):
        super(PDDModel, self).__init__()

        self.model = mobilenet_v2(pretrained)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(1280 * 1 * 1, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        alpha = 10
        self.features = self.features * alpha

        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


def get_trained_model(model, feature_extractor, device):
    model.load_state_dict(
        torch.load(
            feature_extractor,
            map_location=device))
    model.eval()
    return model


