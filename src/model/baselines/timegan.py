"""Translated from Sebastian"""
"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

class TimeGAN(pl.LightningModule):
    def __init__(self, ori_data, parameters):
        super(TimeGAN, self).__init__()
        
        self.ori_data = ori_data
        self.parameters = parameters

        # Parameters
        self.hidden_dim = parameters['hidden_dim']
        self.num_layers = parameters['num_layer']
        self.iterations = parameters['iterations']
        self.batch_size = parameters['batch_size']
        self.z_dim = ori_data.shape[2]  # Assuming `dim` is the last dimension of `ori_data`
        self.gamma = 1
        self.time, self.max_seq_len = self.extract_time(ori_data)

        # Networks
        self.embedder = self._build_rnn(self.hidden_dim, self.num_layers, output_dim=self.hidden_dim)
        self.recovery = self._build_rnn(self.hidden_dim, self.num_layers, output_dim=ori_data.shape[2])
        self.generator = self._build_rnn(self.hidden_dim, self.num_layers, output_dim=self.hidden_dim)
        self.supervisor = self._build_rnn(self.hidden_dim, self.num_layers - 1, output_dim=self.hidden_dim)
        self.discriminator = self._build_rnn(self.hidden_dim, self.num_layers, output_dim=1)

    def _build_rnn(self, hidden_dim, num_layers, output_dim):
        """Helper function to build RNN blocks."""
        layers = [
            nn.RNN(input_size=hidden_dim if i > 0 else self.z_dim,  # Input size for the first layer
                   hidden_size=hidden_dim,
                   num_layers=1,
                   batch_first=True) for i in range(num_layers)
        ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, X, T, Z):
        # Embedder and Recovery
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        # Generator and Supervisor
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        H_hat_supervise = self.supervisor(H)

        # Synthetic Data
        X_hat = self.recovery(H_hat)

        # Discriminator
        Y_fake = self.discriminator(H_hat)
        Y_real = self.discriminator(H)
        Y_fake_e = self.discriminator(E_hat)

        return X_tilde, H_hat, H_hat_supervise, X_hat, Y_fake, Y_real, Y_fake_e

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, T = batch
        Z = self.random_generator(len(X), self.z_dim, T, self.max_seq_len)

        # Forward pass
        X_tilde, H_hat, H_hat_supervise, X_hat, Y_fake, Y_real, Y_fake_e = self(X, T, Z)

        # Losses
        if optimizer_idx == 0:  # Embedder
            E_loss_T0 = nn.MSELoss()(X, X_tilde)
            E_loss0 = 10 * torch.sqrt(E_loss_T0)
            G_loss_S = nn.MSELoss()(H_hat_supervise[:, :-1, :], H_hat[:, 1:, :])
            E_loss = E_loss0 + 0.1 * G_loss_S
            return E_loss

        elif optimizer_idx == 1:  # Discriminator
            D_loss_real = nn.BCEWithLogitsLoss()(Y_real, torch.ones_like(Y_real))
            D_loss_fake = nn.BCEWithLogitsLoss()(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = nn.BCEWithLogitsLoss()(Y_fake_e, torch.zeros_like(Y_fake_e))
            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
            return D_loss

        elif optimizer_idx == 2:  # Generator
            G_loss_U = nn.BCEWithLogitsLoss()(Y_fake, torch.ones_like(Y_fake))
            G_loss_U_e = nn.BCEWithLogitsLoss()(Y_fake_e, torch.ones_like(Y_fake_e))

            G_loss_V1 = torch.mean(torch.abs(
                torch.sqrt(torch.var(X_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(X, dim=0) + 1e-6)))
            G_loss_V2 = torch.mean(torch.abs(torch.mean(X_hat, dim=0) - torch.mean(X, dim=0)))

            G_loss_V = G_loss_V1 + G_loss_V2
            G_loss_S = nn.MSELoss()(H_hat_supervise[:, :-1, :], H_hat[:, 1:, :])

            G_loss = G_loss_U + self.gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
            return G_loss

    def configure_optimizers(self):
        optimizer_E = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=0.001)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.001)
        optimizer_G = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=0.001)
        return [optimizer_E, optimizer_D, optimizer_G]

    def random_generator(self, batch_size, z_dim, T_mb, max_seq_len):
        """Random vector generation."""
        Z_mb = []
        for i in range(batch_size):
            temp = np.zeros([max_seq_len, z_dim])
            temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
            temp[:T_mb[i], :] = temp_Z
            Z_mb.append(temp_Z)
        return torch.tensor(Z_mb, dtype=torch.float32)

    def extract_time(self, data):
        """Returns Maximum sequence length and each sequence length."""
        time = []
        max_seq_len = 0
        for i in range(len(data)):
            max_seq_len = max(max_seq_len, len(data[i][:, 0]))
            time.append(len(data[i][:, 0]))
        return time, max_seq_len

    def batch_generator(self, data, time, batch_size):
        """Mini-batch generator."""
        no = len(data)
        idx = np.random.permutation(no)
        train_idx = idx[:batch_size]     
        
        X_mb = [data[i] for i in train_idx]
        T_mb = [time[i] for i in train_idx]
        
        return torch.tensor(X_mb, dtype=torch.float32), torch.tensor(T_mb, dtype=torch.int32)

    def train_test_divide(self, data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
        """Divide train and test data for both original and synthetic data."""
        # Divide train/test index (original data)
        no = len(data_x)
        idx = np.random.permutation(no)
        train_idx = idx[:int(no * train_rate)]
        test_idx = idx[int(no * train_rate):]
        
        train_x = [data_x[i] for i in train_idx]
        test_x = [data_x[i] for i in test_idx]
        train_t = [data_t[i] for i in train_idx]
        test_t = [data_t[i] for i in test_idx]      
        
        # Divide train/test index (synthetic data)
        no = len(data_x_hat)
        idx = np.random.permutation(no)
        train_idx = idx[:int(no * train_rate)]
        test_idx = idx[int(no * train_rate):]
        
        train_x_hat = [data_x_hat[i] for i in train_idx]
        test_x_hat = [data_x_hat[i] for i in test_idx]
        train_t_hat = [data_t_hat[i] for i in train_idx]
        test_t_hat = [data_t_hat[i] for i in test_idx]
        
        return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat

    def rnn_cell(self, module_name, hidden_dim):
        """Basic RNN Cell."""
        assert module_name in ['gru', 'lstm', 'lstmLN']

        if module_name == 'gru':
            return nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif module_name == 'lstm':
            return nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Layer normalization for LSTM is not supported in PyTorch yet.")
