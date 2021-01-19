import numpy as np
import scipy.sparse as sp
import h5py
import matplotlib.pyplot as plt


# import matlab files in python
def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


def load_data():
    path_dataset = 'D:/ykx/GANGMC-GCN-1027/GANGMC-movielens/Data/movielens/split_1.mat'

    # loading of the required matrices
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    Wrow = load_matlab_file(path_dataset, 'W_users')  # sparse
    Wcol = load_matlab_file(path_dataset, 'W_movies')  # sparse

    np.random.seed(0)
    pos_tr_samples = np.where(Otraining)

    num_tr_samples = len(pos_tr_samples[0])
    list_idx = list(range(num_tr_samples))  # differs from range(...) in Python 2 version
    np.random.shuffle(list_idx)
    idx_data = list_idx[:num_tr_samples // 2]
    idx_train = list_idx[num_tr_samples // 2:]
    pos_data_samples = (pos_tr_samples[0][idx_data], pos_tr_samples[1][idx_data])
    pos_tr_samples = (pos_tr_samples[0][idx_train], pos_tr_samples[1][idx_train])

    """
    Odata is used for initializing H and W matrices
    """
    Odata = np.zeros(M.shape)
    for k in range(len(pos_data_samples[0])):
        Odata[pos_data_samples[0][k], pos_data_samples[1][k]] = 1

    Otraining = np.zeros(M.shape)
    for k in range(len(pos_tr_samples[0])):
        Otraining[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1

    print('Num data samples: %d' % (np.sum(Odata),))
    print('Num train samples: %d' % (np.sum(Otraining),))
    print('Num train+data samples: %d' % (np.sum(Odata + Otraining),))

    # computation of the normalized laplacians
    """
    Get the normailized Laplacian matrcies for row (W) and column (H) graphs respectively
    """

    # apply SVD initially for detecting the main components of our initialization
    U, s, V = sp.linalg.svds(Odata * M, k=10)

    rank_W_H = 10
    partial_s = s[:rank_W_H]
    partial_S_sqrt = np.diag(np.sqrt(partial_s))
    initial_H = np.dot(U[:, :rank_W_H], partial_S_sqrt)
    initial_W = np.dot(partial_S_sqrt, V[:rank_W_H, :]).T

    print(initial_W.shape)
    print(initial_H.shape)

    H_edge_index = sp.coo_matrix(Wrow)
    W_edge_index = sp.coo_matrix(Wcol)

    M = sp.coo_matrix(M)

    return initial_H, initial_W, H_edge_index, W_edge_index, M, Otraining, Odata, Otest


# Plot losses
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='MNIST_DCGAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epoch)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

