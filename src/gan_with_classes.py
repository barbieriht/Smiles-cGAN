import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.autograd as autograd

import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

from visualize_statistics import plot

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# '''Reference: https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0#:~:text=A%20straight%2Dthrough%20estimator%20is,function%20was%20an%20identity%20function.'''

# class STEFunction(autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0).float()
#     @staticmethod
#     def backward(ctx, grad_output):
#         return F.hardtanh(grad_output)
# class StraightThroughEstimator(nn.Module):
#     def __init__(self):
#         super(StraightThroughEstimator, self).__init__()
#     def forward(self, x):
#         x = STEFunction.apply(x)
#         return x
    
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

def preprocessing_data(molecules, replacement, saving=False):

    if not saving:
        molecules = pd.Series(molecules)
    else:
        molecules = pd.Series([mol for mol in molecules])

    for pattern, repl in replacement.items():
        molecules = molecules.str.replace(pattern, repl, regex=False)

    max_length = len(max(molecules, key=len))

    return molecules, max_length

def get_hot_smiles(file_name):
    with open(file_name, "r") as f:
        smiles = [i.strip().split(' ') for i in f.readlines()]
        f.close()

    ########## CLASSES ########## 
    classes = []
    for s in smiles:
        classes.append(s[1].split("@"))

    unique_elements = [''] + list(set([item for sublist in classes for item in sublist]))

    classes_data = []
    for item in classes:
        this_row = np.array([])
        for classe in item:
            this_row = np.append(this_row, unique_elements.index(classe))
        classes_data.append(this_row)

    max_length = max(len(arr) for arr in classes_data)
    classes_hot_arrays = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=0) for arr in classes_data])

    ########## MOLECULES ##########
    molecules = []
    for s in smiles:
        molecules.append(s[0])

    processed_molecules, max_length = preprocessing_data(molecules, replace_dict)

    coder = smiles_coder()
    coder.fit(processed_molecules, max_length)
    smiles_hot_arrays = coder.transform(processed_molecules)

    return smiles_hot_arrays, molecules, classes_hot_arrays, coder, unique_elements

class DrugLikeMolecules(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.smiles, self.dataset, self.classes, self.coder, self.classes_code = get_hot_smiles("chebi_smiles_1of10_subset.txt")
        
        self.smiles_nodes = self.smiles.shape[1] * self.smiles.shape[2]
        self.classes_nodes = self.classes.shape[1]
        self.unique_classes = len(self.classes_code)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        sml = self.smiles[idx]
        labels = self.classes[idx]

        if self.transform:
            sml = self.transform(sml)

        return sml, labels
    
############################# SIMPLE DISCRIMINATOR #######################
# class Discriminator(nn.Module):
#     def __init__(self, smiles_nodes, smiles_shape):
#         super().__init__()
#         self.smiles_nodes = smiles_nodes
#         self.smiles_shape = smiles_shape

#         self.model = nn.Sequential(
#             nn.Linear(self.smiles_nodes, 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#       x = x.view(x.size(0), self.smiles_nodes)

#       x = torch.cat([x], 1)
#       out = self.model(x.float())
#       return out.squeeze()
    
############################# ENSEMBLE DISCRIMINATOR #######################
class Generator(nn.Module):
    def __init__(self, smiles_nodes, smiles_shape, classes_nodes, classes_shape, unique_classes):
        super().__init__()
        self.smiles_nodes = smiles_nodes
        self.smiles_shape = smiles_shape

        self.classes_nodes = classes_nodes
        self.classes_shape = classes_shape
        self.unique_classes = unique_classes

        self.label_emb = nn.Embedding(self.unique_classes, self.classes_nodes)

        self.model = nn.Sequential(
            nn.Linear(100 + self.classes_nodes**2, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, smiles_nodes),
            nn.Tanh()
        )

    def forward(self, z, c):
        z = z.view(z.size(0), 100)
        c = c.view(c.size(0), self.classes_nodes)

        assert torch.all(c >= 0)
        assert torch.all(c < self.label_emb.num_embeddings)

        c = self.label_emb(c)

        c = c.view(c.size(0), -1)

        x = torch.cat([z, c], 1)
        smile_out = self.model(x)
        
        return smile_out.view(-1, self.smiles_nodes)

class Discriminator(nn.Module):
    def __init__(self, smiles_nodes, smiles_shape, classes_nodes, classes_shape, unique_classes):
        super().__init__()
        self.smiles_nodes = smiles_nodes
        self.smiles_shape = smiles_shape
        
        self.classes_nodes = classes_nodes
        self.classes_shape = classes_shape
        self.unique_classes = unique_classes

        self.label_emb = nn.Embedding(self.unique_classes, self.classes_nodes)

        self.smile_input = nn.Sequential(
            nn.Linear(self.smiles_nodes + self.classes_nodes**2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        self.parallel1_ = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

        )

        self.parallel_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.parallel2_ = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        self.parallel_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.parallel_3 = nn.Sequential(
            nn.Linear(256, 1)
        )

        self.final = nn.Sequential(
            nn.Linear(3, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        x = x.view(x.size(0), self.smiles_nodes)
        c = c.view(c.size(0), self.classes_nodes)

        # print(f"label_emb.num_embeddings: {self.label_emb.num_embeddings}")
        # print(f"unique_classes: {self.unique_classes}; classes_nodes: {self.classes_nodes}")

        # assert torch.all(c >= 0)
        # assert torch.all(c < self.label_emb.num_embeddings)

        c = self.label_emb(c)

        c = c.view(c.size(0), -1)

        x = torch.cat([x, c], 1)

        x0 = self.smile_input(x.to(torch.float).to(device))

        x_1 = self.parallel_1(x0)
        x1_ = self.parallel1_(x0)

        x_2 = self.parallel_2(x1_)
        x2_ = self.parallel2_(x1_)

        x_3 = self.parallel_3(x2_)

        x_concatenated = torch.cat([x_1, x_2, x_3], 1)
        
        final_x = self.final(x_concatenated)

        out = self.sigmoid(final_x)
    
        return out.squeeze()
    
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, labels_shape, num_classes):
    g_optimizer.zero_grad()
    
    z = Variable(torch.randn(batch_size, 100)).to(device)
    # c = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.randint(0, num_classes, size=(labels_shape)).to(torch.long)).to(device)

    fake_smiles = generator(z, fake_labels)
    validity = discriminator(fake_smiles, fake_labels)

    g_loss = criterion(validity, Variable(torch.ones(batch_size).to(torch.float)).to(device).float())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_smiles, labels, num_classes):
    d_optimizer.zero_grad()

    # train with real smiles

    real_validity = discriminator(real_smiles, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

    # train with fake smiles
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.randint(0, num_classes, size=(labels.shape)).to(torch.long)).to(device)

    fake_smiles = generator(z, fake_labels)

    fake_validity = discriminator(fake_smiles, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item(), real_loss.data.item(),  fake_loss.data.item()

def save_smiles(epoch, sample_smiles, sample_classes, dataset, params):

    if not os.path.exists('generated_files'):
        os.mkdir('generated_files')

    for i in ("smiles", "classes"):
        if not os.path.exists(f"./generated_files/{i}_lr{params['learning_rate']}_bpe{params['batch_per_epoca']}"):
            os.mkdir(f"./generated_files/{i}_lr{params['learning_rate']}_bpe{params['batch_per_epoca']}")

    # Translating smiles
    all_gen_smiles = []
    for sml in sample_smiles:
        this_smile = np.reshape(sml.detach().cpu().numpy(), (1, dataset.smiles.shape[1], dataset.smiles.shape[2]))
        all_gen_smiles.append(dataset.coder.inverse_transform(this_smile)[0])
    processed_molecules = preprocessing_data(all_gen_smiles, inverse_dict)

    # Translating classes
    all_gen_classes = []
    for cls in sample_classes:
        this_classes = [dataset.classes_code[pos] for pos in cls.detach().cpu().numpy()]
        all_gen_classes.append(this_classes)

    with open(f"./generated_files/smiles_lr{params['learning_rate']}_bpe{params['batch_per_epoca']}/cgan_{epoch}.txt", 'a') as f:
        for molecule in processed_molecules[0]:
            f.write(molecule)
            f.write('\n')
        f.close()

    with open(f"./generated_files/classes_lr{params['learning_rate']}_bpe{params['batch_per_epoca']}/cgan_{epoch}.txt", 'a') as f:
        for classes in all_gen_classes:
            f.write('@'.join(classes))
            f.write('\n')
        f.close()

def train(params, criterion, batch_size=None, num_epochs = 500, display_step = 10, num_classes = 150):

    train_tracking = {}

    d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=params['learning_rate'])
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=params['learning_rate'])
    
    for epoch in range(num_epochs):
        for i, (smiles, labels) in enumerate(data_loader):
            print('Training model >> Epoch: [{}] -- Batch: [{}]'.format(epoch, i), end='\r')

            if len(smiles) != batch_size or len(labels) != batch_size:
                continue

            real_smiles = Variable(smiles.to(torch.float)).to(device).squeeze(1)
            labels = Variable(labels.to(torch.long)).to(device).squeeze(1)
            
            assert (labels<num_classes).all(), "target: {} invalid".format(labels)

            generator.train()

            if batch_size == None:
                batch_size = real_smiles.size(0)
                

            d_loss, d_real_loss, d_fake_loss = discriminator_train_step(batch_size, discriminator,
                                                generator, d_optimizer, criterion, real_smiles,
                                                labels, num_classes)

            g_loss  = generator_train_step(batch_size, discriminator, generator, g_optimizer,
                                           criterion, labels.shape, num_classes)

        generator.eval()
        print('Saving smiles >> Epoch: [{}] --- g_loss: {}, d_loss: {}'.format(epoch, g_loss, d_loss))

        if epoch % display_step == 0:
            z = Variable(torch.randn(32, 100)).to(device)
            sample_classes = Variable(torch.randint(0, num_classes, size=(32, dataset.classes.shape[1])).to(torch.long)).to(device)
            sample_smiles = generator(z, sample_classes)
            save_smiles(epoch, sample_smiles, sample_classes, dataset, params)

        train_tracking[epoch] = {"Discriminator Total Loss":d_loss, "Discriminator Real Loss":d_real_loss,
                                 "Discriminator Fake Loss":d_fake_loss, "Generator Loss":g_loss}

        

if __name__ == "__main__":
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])  
    dataset = DrugLikeMolecules(transform=transform)
 
    criterion = nn.BCELoss()
    generator = Generator(dataset.smiles_nodes, dataset.smiles.shape, dataset.classes_nodes, dataset.classes.shape, dataset.unique_classes).to(device)
    discriminator = Discriminator(dataset.smiles_nodes, dataset.smiles.shape, dataset.classes_nodes, dataset.classes.shape, dataset.unique_classes).to(device)

    params = {"learning_rate": [0.01, 0.001, 0.0001],
              "batch_per_epoca": [12, 24, 48]}
    
    combinations = list(product(*params.values()))

    for param in combinations:
        this_params = dict(zip(params.keys(), param))

        print(this_params)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=this_params['batch_per_epoca'], shuffle=True)

        statistics = train(this_params, criterion, batch_size=this_params['batch_per_epoca'], num_classes=dataset.unique_classes)

        plot(statistics, batch_per_epoca=this_params["batch_per_epoca"], learning_rate=this_params["learning_rate"])
    