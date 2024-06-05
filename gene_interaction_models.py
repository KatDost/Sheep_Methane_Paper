# Source:
## Cui, T., El Mekkaoui, K., Reinvall, J. et al. Gene–gene 
## interaction detection with deep learning. Commun Biol 5, 1238 
## (2022). https://doi.org/10.1038/s42003-022-04186-y

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):     
        super(LinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.w = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))

    def forward(self, x):
        output = torch.mm(x.double(), self.w.double()) + self.bias          
        return output
    
class SparseLinearLayer(nn.Module):
    def __init__(self, gene_size, device):     
        super(SparseLinearLayer, self).__init__()
        self.device = device
        self.input_dim = sum(gene_size)
        self.output_dim = len(gene_size)
        self.mask = self._mask(gene_size).detach().to(self.device)
        
        scale = 1. * np.sqrt(6. / (self.input_dim + self.output_dim))
        # approximated posterior
        self.w = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale).to(self.device) * self.mask) 
        self.bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale).to(self.device))

    def forward(self, x):
        output = torch.mm(x, self.w * self.mask) + self.bias          
        return output
    
    def _mask(self, gene_size):
        index_gene = []
        index_gene.append(0)
        for i in range(len(gene_size)):
            index_gene.append(gene_size[i] + index_gene[i])
        sparse_mask = torch.zeros(sum(gene_size), len(gene_size))
        for i in range(len(gene_size)):
            sparse_mask[index_gene[i]:index_gene[i+1], i]=1
        return sparse_mask


# class Encoder(nn.Module):
#     def __init__(self, gene_size, device):
#         super(Encoder, self).__init__()
#         self.layer = SparseLinearLayer(gene_size, device)
        
#     def forward(self, x):
#         x = self.layer(x)
#         return x, self.reg_layers()
    
#     def reg_layers(self):
#         reg = torch.norm(self.layer.w, 1)
#         return reg 

class Encoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, device):
        super(Encoder, self).__init__()
        self.layer = LinearLayer(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer(x)
        return x, self.reg_layers()
    
    def reg_layers(self):
        reg = torch.norm(self.layer.w, 1)
        return reg 
    
# class Predictor(nn.Module):
#     def __init__(self, gene_size):
#         super(Predictor, self).__init__()
#         self.input_dim = len(gene_size)
        
#         self.Layer1 = LinearLayer(self.input_dim, 100)
#         self.Layer2 = LinearLayer(100, 1)
#         self.activation_fn = nn.Softplus(beta = 10)
        
#     def forward(self, x):
#         x1 = self.activation_fn(self.Layer1(x))
#         x2 = self.Layer2(x1)

#         return x2, self.reg_layers()
    
#     def reg_layers(self):
#         reg = torch.norm(self.Layer1.w, 1) + torch.norm(self.Layer2.w, 1)
#         return reg 
    
class Predictor(nn.Module):
    def __init__(self, pred_dim):
        super(Predictor, self).__init__()
        self.input_dim = pred_dim
        
        self.Layer1 = LinearLayer(self.input_dim, 100)
        self.Layer2 = LinearLayer(100, 1)
        self.activation_fn = nn.Softplus(beta = 10)
        
    def forward(self, x):
        x1 = self.activation_fn(self.Layer1(x))
        x2 = self.Layer2(x1)

        return x2, self.reg_layers()
    
    def reg_layers(self):
        reg = torch.norm(self.Layer1.w, 1) + torch.norm(self.Layer2.w, 1)
        return reg 
        

class Main_effect(nn.Module):
    def __init__(self, gene_size):
        super(Main_effect, self).__init__()
        self.input_dim = len(gene_size)
        self.Layer1 = LinearLayer(self.input_dim, 1)
        
    def forward(self, x):
        x = self.Layer1(x)
        return x, self.reg_layers()
    
    def reg_layers(self):
        reg = torch.norm(self.Layer1.w, 1)
        return reg 
    
class SparseNN(nn.Module):
    def __init__(self, encoder, predictor):
        super(SparseNN, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        
    def forward(self, x):
        x1, kl1 = self.encoder(x)
        x2, kl2 = self.predictor(x1)

        return x2, kl1, kl2    


class NNtraining(object):
    def __init__(self, 
                 model, 
                 learning_rate=0.001, 
                 batch_size=10000, 
                 num_epoch=200, 
                 early_stop_patience = 20,
                 reg_weight_encoder = 0.0,
                 reg_weight_predictor = 0.0,
                 use_cuda=False,
                 use_early_stopping = False):
        
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.best_val = 1e5
        self.early_stop_patience = early_stop_patience
        self.epochs_since_update = 0  # used for early stopping
        self.reg_weight_encoder = reg_weight_encoder
        self.reg_weight_predictor = reg_weight_predictor
        self.use_early_stopping = use_early_stopping
        
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        
    # def training(self, x, y, xval, yval):
  
    #     parameters = set(self.model.parameters())
    #     optimizer = optim.Adam(parameters, lr=self.learning_rate, eps=1e-3)
    #     criterion = nn.MSELoss()
    #     if self.use_cuda:
    #         x = x.cuda()
    #         y = y.cuda()
    #     train_dl = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)       
    #     for epoch in range(self.num_epoch):
    #         for x_batch, y_batch in train_dl:
    #             optimizer.zero_grad()
    #             self.model.train()
    #             # calculate the training loss
    #             output, reg_encoder, reg_predictor = self.model(x_batch)
    #             loss = criterion(y_batch, output) + self.reg_weight_encoder * reg_encoder + self.reg_weight_predictor * reg_predictor
    #             # backpropogate the gradient
    #             loss.backward()
    #             # optimize with SGD
    #             optimizer.step()
            
    #         train_mse, train_pve = self.build_evaluation(x, y)
    #         val_mse, val_pve = self.build_evaluation(xval, yval)
    #         print('>>> Epoch {:5d}/{:5d} | train_mse={:.5f} | val_mse={:.5f} | train_pve={:.5f} | val_pve={:.5f}'.format(epoch,
    #                                                                                                                      self.num_epoch, 
    #                                                                                                                      train_mse, 
    #                                                                                                                      val_mse, 
    #                                                                                                                      train_pve, 
    #                                                                                                                      val_pve))
    #         if self.use_early_stopping:
    #             early_stop = self._early_stop(val_mse)
    #             if early_stop:
    #                 break

    def training(self, x, y, xval, yval):
        losses = []
        validation_losses = []
        parameters = set(self.model.parameters())
        optimizer = optim.Adam(parameters, lr=self.learning_rate, eps=1e-3)
        criterion = nn.MSELoss()
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        train_dl = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)       
        for epoch in range(self.num_epoch):
            for x_batch, y_batch in train_dl:
                optimizer.zero_grad()
                self.model.train()
                # calculate the training loss
                output, reg_encoder, reg_predictor = self.model(x_batch)
                loss = criterion(y_batch, output) + self.reg_weight_encoder * reg_encoder + self.reg_weight_predictor * reg_predictor
                # backpropogate the gradient
                loss.backward()
                # optimize with SGD
                optimizer.step()
            
            train_mse, train_pve = self.build_evaluation(x, y)
            val_mse, val_pve = self.build_evaluation(xval, yval)
            losses.append(train_mse.item())
            validation_losses.append(val_mse.item())
            
            if self.use_early_stopping:
                early_stop = self._early_stop(val_mse)
                if early_stop:
                    break
        return losses, validation_losses
    
                
    def build_evaluation(self, x_test, y_test):
        criterion = nn.MSELoss()
        if self.use_cuda:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        self.model.eval()
        y_pred, _, _ = self.model(x_test)
        mse_eval = criterion(y_test, y_pred).detach()
        
        pve = (1. - torch.var(y_pred.view(-1) - y_test.view(-1)) / torch.var(y_test.view(-1))).detach() 
        return mse_eval, pve
    
    def _early_stop(self, val_loss):
        updated = False # flag
        current = val_loss
        best = self.best_val
        improvement = (best - current) / best
#         improvement  = best - current
        
        if improvement > 0.00:
            self.best_val = current
            updated = True
        
        if updated:
            self.epochs_since_update = 0
        else:
            self.epochs_since_update += 1
            
        return self.epochs_since_update > self.early_stop_patience

def load_data_permutation(x_path, y_path):
    Y = np.genfromtxt(y_path, delimiter=",")
    X = np.genfromtxt(x_path, delimiter=",")

    ## scale each SNPs to have unit variance
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - np.mean(X[:,i])

        Y = (Y - np.mean(Y)) / np.std(Y)
    # seperate training and testing data
    np.random.seed(129)
    msk = np.random.rand(len(X)) < 0.7
    x_train = X[msk,:]; x_test = X[~msk,:]
    y_train = Y[msk]; y_test = Y[~msk]

    x_train = torch.tensor(x_train, dtype = torch.float)
    x_test = torch.tensor(x_test, dtype = torch.float)
    y_train = torch.tensor(y_train, dtype = torch.float)
    y_test = torch.tensor(y_test, dtype = torch.float)
    ## consider the first phenotype
    phenotype_y = y_train.view(-1,1)
    phenotype_y_test = y_test.view(-1,1)
    return x_train, x_test, phenotype_y, phenotype_y_test, torch.tensor(X, dtype = torch.float), torch.tensor(Y, dtype = torch.float)


def preprocessing_permutation(X, Y):
    X_prep = X; Y_prep = Y
    ## scale each SNPs to have unit variance
    for i in range(X.shape[1]):
        X_prep[:,i] = X[:,i] - np.mean(X[:,i])   

        Y = (Y - np.mean(Y)) / np.std(Y)

    # seperate training and testing data
    np.random.seed(129)
    msk = np.random.rand(len(X_prep)) < 0.7
    x_train = X_prep[msk,:]; x_test = X_prep[~msk,:]
    y_train = Y_prep[msk,:]; y_test = Y_prep[~msk,:]

    x_train = torch.tensor(x_train, dtype = torch.float); x_test = torch.tensor(x_test, dtype = torch.float)
    y_train = torch.tensor(y_train, dtype = torch.float); y_test = torch.tensor(y_test, dtype = torch.float)
    return x_train, x_test, y_train, y_test, torch.tensor(X_prep, dtype = torch.float), torch.tensor(Y_prep, dtype = torch.float)



## Interaction detection scores
def matric2dic(hessian, K):
    IS = {}
    for i in range(len(hessian[0])):
        for j in range(i+1, len(hessian[0])):
            tmp = 0
            interation = 'Interaction: '
            interation = interation + str(i + 1) + ' ' + str(j + 1) + ' '
            IS[interation] = hessian[i][j]
    Sorted_IS = [(k, IS[k]) for k in sorted(IS, key=IS.get, reverse=True)]
    return IS, Sorted_IS

def inputGradient(predictor, x):
    output, _ = predictor(x)
    first = torch.autograd.grad(output, x)
    return first[0].view(-1)

def inputHessian(predictor, x, device):
    Hessian = []
    output, _ = predictor(x)
    first = torch.autograd.grad(output, x, create_graph=True)
    num_gene = x.shape[1]
    for i in range(num_gene):
        gradient = torch.zeros(num_gene, dtype = torch.float).to(device)
        gradient[i] = 1.0
        second = torch.autograd.grad(first, x, grad_outputs=gradient.view(1,-1), retain_graph=True)
        Hessian.append(second[0][0].tolist())
    return Hessian

def IntegratedHessian(predictor, xi, baseline, device):
    num_gene = xi.shape[1]; m = 5; k = 5
    diff = xi - baseline
    Diff2 = torch.ger(diff.view(-1), diff.view(-1))
    PathHessian = torch.zeros([num_gene, num_gene]).to(device)
    PathGradient = torch.zeros([num_gene]).to(device)
    # discrete path integral
    for p in range(m):
        for l in range(k):
            x_eva = (baseline + (l+1) / k * (p+1) / m * diff).requires_grad_(True)
            PathHessian = PathHessian + (l+1) / k * (p+1) / m * torch.tensor(inputHessian(predictor, x_eva, device)).to(device) / (k * m)
            PathGradient = PathGradient + inputGradient(predictor, x_eva) / (k * m)
    ItgHessian = PathHessian * Diff2 + torch.diag(PathGradient * diff.view(-1))
    return ItgHessian

def GlobalIH(predictor, X, baseline, device):
    num_individual, num_gene = X.shape
    Hessian = torch.zeros([num_gene, num_gene]).to(device)
    
    for i in range(num_individual):
        x = X[i].clone().view(1,-1)
        Hessian = Hessian + torch.abs(IntegratedHessian(predictor, x, baseline, device))
    Hessian = Hessian / num_individual
    GlobalIH, topGlobalIH = matric2dic(Hessian, 10)
    return GlobalIH, topGlobalIH, Hessian

def copy_values(xi, baseline, index_set):
    tij = baseline.clone()
    for i in index_set:
        tij[i] = xi[i]
    return tij

def delta_main(predictor, xi, baseline, main_index):
    Ti = copy_values(xi, baseline, main_index).view(1,-1)
    T = copy_values(xi, baseline, []).view(1,-1)
    output_Ti, _ = predictor(Ti); output_T, _ = predictor(T); 
    return output_Ti.item() - output_T.item()

def deltaF(predictor, xi, baseline, interaction, T):
    Tij = copy_values(xi, baseline, T + interaction).view(1,-1)
    Ti = copy_values(xi, baseline, T + [interaction[0]]).view(1,-1)
    Tj = copy_values(xi, baseline, T + [interaction[1]]).view(1,-1)
    T = copy_values(xi, baseline, T).view(1,-1)
    output_Tij, _ = predictor(Tij); output_Ti, _ = predictor(Ti); output_Tj, _ = predictor(Tj); output_T, _ = predictor(T); 
    return output_Tij.item() - output_Ti.item() - output_Tj.item() + output_T.item()

def ShapleyValue(predictor, xi, baseline):
    num_gene = xi.shape[0]
    shapleyvalue = np.zeros([num_gene])
    for i in range(num_gene):
        shapleyvalue[i] = delta_main(predictor, xi, baseline, [i])
    return shapleyvalue

def ShapleyIS(predictor, xi, baseline, num_permutation):
    num_gene = xi.shape[0]
    SHAPLEYIS = np.zeros([num_gene, num_gene])
    for m in range(num_permutation):
        perm = list(np.random.permutation(num_gene)); T = []
        shapleyis = np.zeros([num_gene, num_gene])
        for i in range(len(perm)):
            if i >= 1:
                T.append(perm[i-1])
            for j in range(i+1, len(perm)):
                shapleyis[perm[i]][perm[j]] = deltaF(predictor, xi, baseline, [perm[i],perm[j]], T)
        SHAPLEYIS = SHAPLEYIS + shapleyis

    SHAPLEYIS = (SHAPLEYIS + SHAPLEYIS.T) / num_permutation
    SHAPLEYIS = SHAPLEYIS +  np.diag(ShapleyValue(predictor, xi, baseline)) 
    return SHAPLEYIS

def GlobalSIS(predictor, X, baseline, num_permutation = 10):
    num_individual, num_gene = X.shape
    Shapely = np.zeros([num_gene, num_gene])    
    for i in range(num_individual):
        x = X[i].clone()
        Shapely = Shapely + abs(ShapleyIS(predictor, x, baseline.view(-1), num_permutation))
    Shapely = Shapely / num_individual
    GlobalSIS, topGlobalSIS = matric2dic(Shapely, 10)
    return GlobalSIS, topGlobalSIS, Shapely 
    