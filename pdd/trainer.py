from .metrics import knn_acc
from .plot import plot
import numpy as np
from scipy.spatial.distance import cosine
import torch
from tqdm import tqdm

COUNT_NEIGHBOR_EXP_1 = 1
COUNT_NEIGHBOR_EXP_2 = 3
METRIC_KNN = cosine
COLORS = ['#00ffff', '#000000', '#0000ff', '#ff00ff',
          '#808080', '#008000', '#00ff00', '#800000',
          '#000080', '#808000', '#800080', '#ff0000',
          '#c0c0c0', '#008080', '#ffff00']


def forward_inputs_into_model(loader, model, device, batch_size):
    X = []
    y = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            if (outputs.shape[0] == batch_size):
                X.append(outputs)
                y.append(targets)
    return np.vstack(X), np.hstack(y)


def save_model(model, optimizer,model_save_path,optim_save_path):
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict, optim_save_path)


class TripletTrainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 train_triplet_loader,
                 epochs,
                 test_triplet_loader,
                 batch_size,
                 knn_train_loader,
                 knn_test_loader,
                 scheduler,
                 nameofplotclasses,
                 num_classes,
                 miner,
                 loss_history,
                 safe_plot_img_path,
                 model_save_path,
                 optim_save_path
                 ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = optimizer
        # self.loss = loss
        self.epochs = epochs
        self.tri_train_load = train_triplet_loader
        self.tri_test_load = test_triplet_loader
        self.batch_size = batch_size
        self.train_loader = knn_train_loader
        self.test_loader = knn_test_loader
#         self.scheduler=scheduler
        self.nameofplotClasses = nameofplotclasses
        self.num_classes = num_classes
        self.miner = miner
        self.loss_history = loss_history
        self.safe_plot_img_path=safe_plot_img_path
        self.model_save_path=model_save_path
        self.optim_save_path=optim_save_path

    def train(self):
        for e in tqdm(range(self.epochs), desc='Epoch'):

            test_em, test_labels = forward_inputs_into_model(self.test_loader, self.model,
                                                             self.device, self.batch_size)
            train_em, train_labels = forward_inputs_into_model(self.train_loader, self.model,
                                                               self.device, self.batch_size)
            knn_acc(
                test_em,
                test_labels,
                train_em,
                train_labels,
                COUNT_NEIGHBOR_EXP_1,
                METRIC_KNN)
            knn_acc(
                test_em,
                test_labels,
                train_em,
                train_labels,
                COUNT_NEIGHBOR_EXP_2,
                METRIC_KNN)

            plot(
                train_em,
                train_labels,
                self.nameofplotClasses,
                'train_embeddings',
                COLORS,
                self.safe_plot_img_path)
            plot(
                test_em,
                test_labels,
                self.nameofplotClasses,
                'test_embeddings',
                COLORS,
                self.safe_plot_img_path)
            self.train_phase()
            self.validating_phase()

            if e % 5 == 0 and e > 0:
                save_model(self.model, self.optimizer,self.model_save_path,self.optim_save_path)

    def train_phase(self):
       
        train_n = len(self.tri_train_load)
        train_loss = 0.
        train_frac_pos = 0.

        self.model.train()
        with tqdm(self.tri_train_load, desc='Batch') as b_pbar:
            for b, batch in enumerate(b_pbar):
                self.optimizer.zero_grad()

                labels, data = batch
                labels = torch.cat([label for label in labels], axis=0)
                data = torch.cat([datum for datum in data], axis=0)
                labels = labels.cuda()
                data = data.cuda()

                embeddings = self.model(data)
                loss, frac_pos = self.miner(labels, embeddings)
                loss.backward()
                self.optimizer.step()
#                     scheduler.step()
                train_loss += loss.detach().item()
                train_frac_pos += frac_pos.detach().item() if frac_pos is not None else \
                    0.

                b_pbar.set_postfix(
                    train_loss=train_loss / train_n,
                    train_frac_pos=f'{(train_frac_pos/train_n):.2%}'
                )

    def validating_phase(self):
        val_n = len(self.tri_test_load)
        val_loss = 0.
        val_frac_pos = 0.
        self.model.eval()
        with tqdm(self.tri_test_load, desc='val') as b_pbar:
            for b, batch in enumerate(b_pbar):
                labels, data = batch
                labels = torch.cat([label for label in labels], axis=0)
                data = torch.cat([datum for datum in data], axis=0)
                labels = labels.cuda()
                data = data.cuda()
                embeddings = self.model(data)
                loss, frac_pos = self.miner(labels, embeddings)
                val_loss += loss.detach().item()
                self.loss_history.append(val_loss)
                val_frac_pos += frac_pos.detach().item() if frac_pos is not None else \
                    0.
                b_pbar.set_postfix(
                    val_loss=val_loss / val_n,
                    val_frac_pos=f'{( val_frac_pos / val_n ):.2%}'
                )
