import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from numpy import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class smiles_coder:
    def __init__(self):
        self.char_set = set([' '])
        self.char_to_int = None
        self.int_to_char = None
        self.fitted = False

    def fit(self, smiles_data, max_length = 150):
        for i in tqdm(range(len(smiles_data))):
            smiles_data[i] = smiles_data[i].ljust(max_length)
            self.char_set = self.char_set.union(set(smiles_data[i]))
        self.max_length = max_length
        self.n_class = len(self.char_set)
        self.char_to_int = dict((c, i) for i, c in enumerate(self.char_set))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.char_set))
        self.fitted = True

    def transform(self, smiles_data):
        if not self.fitted:
            raise ValueError('smiles coder is not fitted')
        m = []
        for i in tqdm(range(len(smiles_data))):
            smiles_data[i] = smiles_data[i].ljust(self.max_length)
            chars = smiles_data[i]
            l = np.zeros((self.max_length, self.n_class))
            for t, char in enumerate(chars):
                if t >= self.max_length:
                    break
                else:
                    if char in self.char_set:
                        l[t, self.char_to_int[char]] = 1
            m.append(l)
        return np.array(m)

    def label(self, smiles_data):
        if not self.fitted:
            raise ValueError('smiles coder is not fitted')
        m = []
        for i in tqdm(range(len(smiles_data))):
            smiles_data[i] = smiles_data[i].ljust(self.max_length)
            chars = smiles_data[i]
            l = np.zeros((self.max_length, 1))
            for t, char in enumerate(chars):
                if t >= self.max_length:
                    break
                else:
                    if char in self.char_set:
                        l[t, 0] = self.char_to_int[char]
            m.append(l)
        return np.array(m)

    def inverse_transform(self, m):
        if not self.fitted:
            raise ValueError('smiles coder is not fitted')
        smiles_out = []
        for l in m:
            ll = np.argmax(l, axis=1)
            string = ''
            for t in ll:
                if self.int_to_char[t] == ' ':
                    continue
                string += self.int_to_char[t]
            smiles_out.append(string)
        return np.array(smiles_out)

    def inverse_label(self, l):
        if not self.fitted:
            raise ValueError('smiles coder is not fitted')
        smiles_out = []
        for ll in l:
            string = ''
            for t in ll:
                if self.int_to_char[t] == ' ':
                    continue
                string += self.int_to_char[t]
            smiles_out.append(string)
        return np.array(smiles_out)

    def int_to_char(self):
        return self.int_to_char

    def char_to_int(self):
        return self.char_to_int

    def save(self, save_path):
        np.savez(save_path, char_set = self.char_set, char_to_int=self.char_to_int, int_to_char=self.int_to_char,
                            max_length = self.max_length, n_class = len(self.char_set))

    def load(self, save_path):
        saved = np.load(save_path, allow_pickle=True)
        self.char_set = saved['char_set'].tolist()
        self.char_to_int = saved['char_to_int'].tolist()
        self.int_to_char = saved['int_to_char'].tolist()
        self.max_length = saved['max_length'].tolist()
        self.n_class = len(self.char_set)
        self.fitted = True

replace_dict = {'Ag':'D', 'Al':'E', 'Ar':'G', 'As':'J', 'Au':'Q', 'Ba':'X', 'Be':'Y',
                'Br':'f', 'Ca':'h', 'Cd':'j', 'Ce':'k', 'Cl':'m', 'Cn':'p', 'Co':'q',
                'Cr':'v', 'Cu':'w', 'Fe':'x', 'Hg':'y', 'Ir':'z', 'La':'!', 'Mg':'$',
                'Mn':'¬', 'Mo':'&', 'Na':'_', 'Ni':'£', 'Pb':'¢', 'Pt':'?', 'Ra':'ª',
                'Ru':'º', 'Sb':';', 'Sc':':', 'Se':'>', 'Si':'<', 'Sn':'§', 'Sr':'~',
                'Te':'^', 'Tl':'|', 'Zn':'{', '@@':'}'}

inverse_dict = {v: k for k, v in replace_dict.items()}

def preprocessing_data(molecules, replacement):
    molecules = pd.Series(molecules)

    for pattern, repl in replacement.items():
        molecules = molecules.str.replace(pattern, repl, regex=False)

    return molecules

def get_hot_smiles(file_name):
    with open(file_name, "r") as f:
        smiles = [i.strip().split(' ') for i in f.readlines()]
        f.close()

    # DEFINING THE CLASSES DATAFRAME
    classes = []
    for s in smiles:
        classes.append(s[1].split("@"))

    unique_elements = list(set([item for sublist in classes for item in sublist]))

    classes_df = pd.DataFrame(columns=unique_elements)

    data = []
    for item in classes:
        row = {element: float(1) if element in item else float(0) for element in unique_elements}
        data.append(row)

    classes_df = pd.concat([classes_df, pd.DataFrame(data)], ignore_index=True)

    classes_hot_arrays = classes_df.values

    molecules = []
    for s in smiles:
        molecules.append(s[0])

    processed_molecules = preprocessing_data(molecules, replace_dict)

    coder = smiles_coder()
    coder.fit(processed_molecules)
    smiles_hot_arrays = coder.transform(processed_molecules)

    return smiles_hot_arrays, molecules, classes_hot_arrays, coder

class DrugLikeMolecules(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.smiles, self.dataset, self.labels, self.coder = get_hot_smiles("chebi_smiles_1of10_subset.txt")

        self.smiles_nodes = self.smiles.shape[1] * self.smiles.shape[2]
        self.classes_nodes = self.labels.shape[1]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sml = self.smiles[idx]

        if self.transform:
            sml = self.transform(sml)

        return sml, label
    
class Discriminator(nn.Module):
    def __init__(self, smiles_nodes, classes_nodes, smiles_shape):
        super().__init__()
        self.smiles_nodes = smiles_nodes
        self.classes_nodes = classes_nodes
        self.smiles_shape = smiles_shape

        self.label_emb = nn.Embedding(self.classes_nodes, self.classes_nodes)

        self.model = nn.Sequential(
            nn.Linear(self.smiles_nodes + self.classes_nodes**2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
      x = x.view(x.size(0), self.smiles_nodes)

      c = labels.view(labels.size(0), -1)
      c = self.label_emb(c)
      c = c.view(c.size(0), -1)

      print(f"X shape: {x.shape} / C shape: {c.shape}")
      x = torch.cat([x, c], 1)
      out = self.model(x)
      return out.squeeze()
    
class Generator(nn.Module):
    def __init__(self, smiles_nodes, classes_nodes, smiles_shape):
        super().__init__()
        self.smiles_nodes = smiles_nodes
        self.classes_nodes = classes_nodes
        self.smiles_shape = smiles_shape

        self.label_emb = nn.Embedding(self.classes_nodes, self.classes_nodes)
        self.input_linear = nn.Sequential(
            nn.Linear(100, 8550),
            nn.LeakyReLU(0.2, inplace=True),)

        self.model = nn.Sequential(
            nn.Linear(self.smiles_nodes + self.classes_nodes**2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, smiles_nodes),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        z = self.input_linear(z)

        c = labels.view(labels.size(0), -1)
        c = self.label_emb(labels)
        c = c.view(c.size(0), -1)

        print(f"z shape: {z.shape} / C shape: {c.shape}")
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, self.smiles_nodes)
    
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, generator.classes_nodes, batch_size))).to(device)
    fake_smiles = generator(z, fake_labels)
    validity = discriminator(fake_smiles, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data[0]

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_smiles, labels):
    d_optimizer.zero_grad()

    # train with real smiles

    real_validity = discriminator(real_smiles, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

    # train with fake smiles
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)

    fake_smiles = generator(z, fake_labels)

    fake_validity = discriminator(fake_smiles, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()

def save_smiles(epoch, sample_smiles):
    # Rescale smiles 0 - 1
    if not os.path.exists('./smiles'):
        os.mkdir('./smiles')

    processed_molecules = preprocessing_data(sample_smiles, inverse_dict)

    with open('smiles/cgan_%d.txt' % epoch, 'a') as f:
        for molecule in processed_molecules:
            f.write(molecule)
            f.write('\n')
        f.close()

if __name__ == "__main__":
    dataset = DrugLikeMolecules()

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])  
    dataset = DrugLikeMolecules(transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle = True)

    generator = Generator(dataset.smiles_nodes, dataset.classes_nodes, dataset.smiles.shape).to(device)
    discriminator = Discriminator(dataset.smiles_nodes, dataset.classes_nodes, dataset.smiles.shape).to(device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    num_epochs = 30
    n_critic = 5
    display_step = 300

    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch))

        for i, (smiles, labels) in enumerate(data_loader):
            real_smiles = Variable(smiles.to(torch.long)).to(device).squeeze(1)
            labels = Variable(labels.to(torch.long)).to(device)

            generator.train()
            batch_size = real_smiles.size(0)

            d_loss = discriminator_train_step(batch_size, discriminator,
                                                generator, d_optimizer, criterion,
                                                real_smiles, labels)

            g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

        generator.eval()
        print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
        z = Variable(torch.randn(9, 100)).cuda()
        labels = Variable(torch.LongTensor(np.arange(9))).cuda()
        sample_smiles = generator(z, labels).unsqueeze(1).data.cpu()
        save_smiles(epoch, sample_smiles)