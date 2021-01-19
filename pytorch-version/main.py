import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
import torch
from torch.autograd import Variable
import Generator_Model as Gen
import Discriminator_Model as Dis
import utils
import scipy.sparse as sp

# torch.cuda.set_device(0)


torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Parameters
D_input_dim = 1
D_output_dim = 1
num_filters = [1024, 512, 256, 1]

G_learning_rate = 0.001
D_learning_rate = 0.0002
betas = (0.5, 0.999)
# batch_size = 128
num_epochs = 100

# load data
initial_H, initial_W, H_edge_index, W_edge_index, M, Otraining, Odata, Otest = utils.load_data()

# Models
G = Gen.Generator(M.shape, factorization_rank=10)
D = Dis.Discriminator(D_input_dim, num_filters[::-1], D_output_dim)
G.cuda()
D.cuda()

# Loss function
criterion = torch.nn.BCELoss()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=G_learning_rate, betas=betas)
D_optimizer = torch.optim.Adam(D.parameters(), lr=D_learning_rate, betas=betas)

# Training GAN
D_avg_losses = []
G_avg_losses = []

RMSE = []

for epoch in range(num_epochs):
    D_losses = []
    G_losses = []

    # Otraining = Otraining + Odata
    M_init = np.multiply((Otraining + Odata), M.todense())
    M_init = sp.coo_matrix(M_init)
    M_init = Gen.tensor_from_scipy_sparse(M_init).cuda()

    x_ = Variable(M_init)  # x_ denotes the real_data
    x_ = torch.unsqueeze(x_, 1)
    x_ = x_.to_dense()

    # labels
    mini_batch = 943
    y_real_ = Variable(torch.ones(mini_batch).cuda())
    y_fake_ = Variable(torch.zeros(mini_batch).cuda())

    # Train discriminator with real data
    D_real_decision = D(x_).squeeze()
    # print(D_real_decision, y_real_)
    D_real_loss = criterion(D_real_decision, y_real_)

    # Train discriminator with fake data
    # generate feak M with inital generator
    epochs = 100
    loss_history = []
    error_history = []
    for i in range(100):
        Hout, Wout, loss_history, error_history = G.train(initial_H, initial_W, H_edge_index, W_edge_index, M, Otraining, Odata, Otest, epochs)
        loss_history += loss_history.tolist()
        error_history += error_history.tolist()
        RMSE.append(min(error_history))
    # print()
    torch.save(G.state_dict(), 'models/model%s_%.4f.pickle' % (epochs * (i + 1), error_history[-1]))
    gen_x = Gen.combine(Hout, Wout)

    gen_x = torch.unsqueeze(gen_x, 1).cuda()
    D_fake_decision = D(gen_x).squeeze()
    D_fake_loss = criterion(D_fake_decision, y_fake_)

    # Back propagation
    D_loss = D_real_loss + D_fake_loss
    D.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    # Train generator with discriminator loss

    # generate feak M with updated generator
    H_temp = torch.tensor(initial_H).float().cuda()
    W_temp = torch.tensor(initial_W).float().cuda()
    H_edge_index_temp = torch.tensor([H_edge_index.row, H_edge_index.col]).long().cuda()
    W_edge_index_temp = torch.tensor([W_edge_index.row, W_edge_index.col]).long().cuda()
    Hout, Wout= G(H_temp, W_temp, H_edge_index_temp, W_edge_index_temp)
    gen_x = Gen.combine(Hout, Wout)
    gen_x = torch.unsqueeze(gen_x, 1).cuda()

    D_fake_decision = D(gen_x).squeeze()
    G_loss = criterion(D_fake_decision, y_real_)

    # Back propagation
    D.zero_grad()
    G.zero_grad()
    # G_loss.backward()
    # G_optimizer.step()

    # loss values
    D_losses.append(D_loss.data)
    G_losses.append(G_loss.data)

    print('Epoch [%d/%d], D_loss: %.4f, G_loss: %.4f' % (epoch + 1, num_epochs, D_loss.data, G_loss.data))

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    utils.plot_loss(D_avg_losses, G_avg_losses, epoch, save=True)

print('RMSE', min(RMSE))




