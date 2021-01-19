import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.sparse as tsps
import torch_geometric.nn as gnn

import scipy.sparse as sp


class Generator(nn.Module):

    def __init__(self, input_shape, factorization_rank=10, n_channels=48,
                 basis_order=5, diffusion_time=10, hidden_cells=48,
                 lstm_layers=1, bidirectional=False):
        super().__init__()

        # GCNN parameters
        self.m = input_shape[0]
        self.n = input_shape[1]
        self.r = factorization_rank
        self.q = n_channels
        self.order = basis_order
        self.loops = True

        # LSTM parameters
        self.nH = hidden_cells
        self.T = diffusion_time
        self.n_layers = lstm_layers
        self.bidirectional = False


        # ChebNet for H, W matrices
        self.hconv = gnn.ChebConv(in_channels=self.r, out_channels=self.q,
                                  K=self.order)
        self.wconv = gnn.ChebConv(in_channels=self.r, out_channels=self.q,
                                  K=self.order)

        # GCN for H, W matrices
        self.hgcn0 = gnn.GCNConv(in_channels=self.r, out_channels=16)
        self.hgcn1 = gnn.GCNConv(in_channels=16, out_channels=self.q)
        self.wgcn0 = gnn.GCNConv(in_channels=self.r, out_channels=16)
        self.wgcn1 = gnn.GCNConv(in_channels=16, out_channels=self.q)


        # RNN
        self.lstm = nn.LSTM(input_size=self.q, hidden_size=self.nH,
                            num_layers=self.n_layers, bidirectional=self.bidirectional,
                            batch_first=True)

        self.dense_H = nn.Linear(in_features=self.nH, out_features=self.r)

        self.dense_W = nn.Linear(in_features=self.nH, out_features=self.r)

        self.loss_fn = recommender_loss

    def init_hidden(self):
        h0 = torch.zeros((self.q,)).view(1, 1, -1)
        c0 = torch.zeros((self.q,)).view(1, 1, -1)
        return h0, c0

    def forward(self, H, W, HA, WA):
        hidden = self.init_hidden()
        """
        diffusion 10 times
        """
        # H = torch.tensor(H).cuda()
        # W = torch.tensor(W).cuda()
        # HA = torch.tensor(HA).cuda()
        # WA = torch.tensor(WA).cuda()
        Hout = H
        Wout = W
        for i in range(self.T):
            hconv0 = self.hgcn0(H, HA)
            hconv1 = self.hgcn1(hconv0, HA)
            Htilde = torch.sigmoid(hconv1)
            out, hidden = self.lstm(Htilde.unsqueeze(0), hidden)
            dout = self.dense_H(out)
            dH = torch.tanh(dout).squeeze()
            Hout = Hout + dH

            wconv0 = self.wgcn0(Wout, WA)
            wconv1 = self.wgcn1(wconv0, WA)
            Wtilde = torch.sigmoid(wconv1)
            out, hidden = self.lstm(Wtilde.unsqueeze(0), hidden)
            dout = self.dense_W(out)
            dW = torch.tanh(dout).squeeze()
            Wout = Wout + dW

        return Hout, Wout

    def train(self, H, W, H_edge_index, W_edge_index, M, Otraining, Odata, Otest, iters, optimizer=None):
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=1e-4)

        training_loss_history = np.zeros((iters,))
        test_RMSE_history = np.zeros((iters,))

        maximum = np.max(M)
        minimum = np.min(M)

        M = sp.coo_matrix(M)
        Otraining = Otraining + Odata
        Otraining = sp.coo_matrix(Otraining)
        Otest = sp.coo_matrix(Otest)
        # Odata = sp.coo_matrix(Odata)

        M = tensor_from_scipy_sparse(M).cuda()
        # Otraining = torch.tensor(Otraining + Odata).cuda()
        # Otest = torch.tensor(Otest).cuda()

        Otraining = tensor_from_scipy_sparse(Otraining).cuda()
        Otest = tensor_from_scipy_sparse(Otest).cuda()

        Lh = sp.csgraph.laplacian(H_edge_index)
        Lw = sp.csgraph.laplacian(W_edge_index)

        Lh = tensor_from_scipy_sparse(Lh).cuda()
        Lw = tensor_from_scipy_sparse(Lw).cuda()

        H = torch.tensor(H).float().cuda()
        W = torch.tensor(W).float().cuda()
        H_edge_index = torch.tensor([H_edge_index.row, H_edge_index.col]).long().cuda()
        W_edge_index = torch.tensor([W_edge_index.row, W_edge_index.col]).long().cuda()

        for i in range(iters):
            Hout, Wout = self.forward(H, W, H_edge_index, W_edge_index)
            loss = self.loss_fn((Hout, Wout), (Lh, Lw), M, Otraining, (minimum, maximum))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss_history[i] = loss.item()
            Xpred = combine(Hout, Wout)
            test_RMSE_history[i] = self.predict(Xpred, M, Otest)
            print('iter %s: loss: %s, error: %s' % (i, training_loss_history[i], test_RMSE_history[i]))

        return Hout.cpu().detach(), Wout.cpu().detach(), training_loss_history, test_RMSE_history

    def predict(self, X, Y, Mask=None):
        X = 1 + 4 * (X - torch.min(X)) / (torch.max(X) - torch.min(X))
        if Mask is None:
            Mask = sp.coo_matrix((np.ones_like(Y.data), (Y.row, Y.col)), shape=Y.shape)
            Mask = tensor_from_scipy_sparse(Mask)
            Y = tensor_from_scipy_sparse(Y)
        predictions = X * Mask.to_dense()
        Y = Mask * Y
        predictions_error = torch.sqrt(torch.sum(torch.square(predictions - Y)) / torch.sum(Mask.to_dense()))
        return predictions_error.item()


def combine(W, H, minimum=1, maximum=5):
    Xpred = torch.mm(W, torch.transpose(H, 0, 1))
    Xpred = minimum + (maximum - 1) * (Xpred - torch.min(Xpred)) / (torch.max(Xpred) - torch.min(Xpred))
    return Xpred


# for computation of loss
def frobenius_norm(x):
    """norm for matrices"""
    x2 = x ** 2
    x2sum = torch.sum(x2)
    return torch.sqrt(x2sum)


def graph_norm(X, laplacian):
    norm = torch.mm(torch.transpose(X, 0, 1), tsps.mm(laplacian, X))
    return norm


def recommender_loss(inputs, laplacians, target, mask, extrema, gamma=1e-10):
    # loss function borrowed from objective in Srebro et. al 2004.
    # set X to valid ratings
    # loss function borrowed from objective in Srebro et. al 2004.
    H, W = inputs
    Lh, Lw = laplacians

    # set X to valid ratings
    X_normed = combine(H, W)

    # consider only original data and test data, ignore other sparse values.
    xm = mask.to_dense() * (X_normed - target)
    fnorm = frobenius_norm(xm)
    fnorm = fnorm / torch.sum(mask.to_dense())

    # compute regularization
    gH = graph_norm(H, Lh)
    gW = graph_norm(W, Lw)
    loss = fnorm + (gamma / 2) * (torch.trace(gH) + torch.trace(gW))
    return loss


def tensor_from_scipy_sparse(X):
    values = X.data
    indices = np.vstack((X.row, X.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    X = torch.sparse.FloatTensor(i, v, torch.Size(X.shape))
    return X

