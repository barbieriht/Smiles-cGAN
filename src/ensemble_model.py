import os
import re
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
import math
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm

from tqdm import tqdm
from itertools import product

from visualize_statistics import plot
from rdkit import Chem
import random

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
    

class smiles_coder:
    def __init__(self):
        self.char_set = set([' '])
        self.char_to_int = None
        self.int_to_char = None
        self.fitted = False

        if os.path.exists('smiles_vocab.npz'):
            self.load('smiles_vocab.npz')

    def fit(self, smiles_data, max_length = 150):
        for i in tqdm(range(len(smiles_data))):
            smiles_data[i] = smiles_data[i] + [" "] * max(0, max_length - len(smiles_data[i]))
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
            smiles_data[i] = smiles_data[i] + [" "] * max(0, self.max_length - len(smiles_data[i]))
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

def smi_tokenizer(smi: str, reverse=False) -> list:
    """
    Tokenize a SMILES molecule
    """
    pattern = r"(\[[^\]]+]|A(?:c|l|m|g|r|s|t|u)?|B(?:a|k|i|h|e|r)?|C(?:a|d|e|f|l|m|n|s|o|r|u)?|D(?:b|s|y)?|E(?:s|r|u)?|F(?:m|l|r|e)?|G(?:d|a|e)?|H(?:f|s|e|o|g)?|I(?:n|r)?|K(?:r)?|L(?:a|r|i|v|u)?|M(?:g|n|o|t|c|d)?|N(?:a|b|d|e|p|h|i|o)?|O(?:g|s)?|P(?:a|b|t|u|o|r|m)?|R(?:a|b|n|e|f|h|g|u)?|S(?:b|c|e|g|i|m|n|r)?|T(?:a|b|c|e|h|i|l|m|s)?|W|U|V|Xe|Y(?:b)?|Z(?:n|r)?|b|c|n|o|s|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|<|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    # tokens = ['<sos>'] + [token for token in regex.findall(smi)] + ['<eos>']
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens[1:-1])
    assert smi == "".join(tokens[:])
    # try:
    #     assert smi == "".join(tokens[:])
    # except:
    #     print(smi)
    #     print("".join(tokens[:]))
    if reverse:
        return tokens[::-1]
    return tokens

def preprocessing_data(molecules, saving=False):
    
    tokenized_molecules = [smi_tokenizer(mol) for mol in molecules]

    if not saving:
        molecules = pd.Series(tokenized_molecules)
    else:
        molecules = pd.Series([mol for mol in tokenized_molecules])

    max_length = len(max(molecules, key=len))

    return molecules, max_length

def get_hot_smiles(file_name):
    with open(file_name, "r") as f:
        smiles = [i.strip().split(' ') for i in f.readlines()]
        f.close()

    ########## CLASSES ########## 
    classes = []
    molecules = []

    for s in smiles:
        for c in s[1].split("@"):
            molecules.append(s[0])
            classes.append(c)

    classes_hot_array = np.array([])

    if not os.path.exists('classes_vocab.json'):
        unique_elements = list(set(classes))

        for classe in classes:
            classes_hot_array = np.append(classes_hot_array, unique_elements.index(classe))

        classes_vocab = {}
        for i, elem in enumerate(unique_elements):
            classes_vocab[elem] = i

        with open('classes_vocab.json', 'w') as f:
            json.dump(classes_vocab, f)
            f.close()
    else:
        with open('classes_vocab.json', 'r') as f:
            classes_vocab = json.load(f)
            f.close()

        for classe in classes:
            classes_hot_array = np.append(classes_hot_array, classes_vocab[classe])

        unique_elements = list(classes_vocab.keys())

    processed_molecules, smiles_max_length = preprocessing_data(molecules)

    ######## TURNING TO ONE HOT ########
    coder = smiles_coder()
    if not os.path.exists('smiles_vocab.npz'):
        coder.fit(processed_molecules, smiles_max_length)
        coder.save('smiles_vocab.npz')
    smiles_hot_arrays = coder.transform(processed_molecules)

    return smiles_hot_arrays, molecules, classes_hot_array, coder, unique_elements

class DrugLikeMolecules(Dataset):
    def __init__(self, transform=None, file_path="chebi_smiles_com_classes.txt"):
        self.transform = transform
        self.smiles, self.dataset, self.classes, self.coder, self.classes_code = get_hot_smiles(file_path)
        
        self.smiles_nodes = self.smiles.shape[1] * self.smiles.shape[2]
        self.unique_classes = len(self.classes_code)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        sml = self.smiles[idx]
        labels = self.classes[idx]

        if self.transform:
            sml = self.transform(sml)

        return sml, labels

class Generator(nn.Module):
    def __init__(self, smiles_nodes, smiles_shape, classes_shape, unique_classes, noise_dim, MIN_DIM):
        super().__init__()
        self.smiles_nodes = smiles_nodes
        self.smiles_shape = smiles_shape

        self.classes_shape = classes_shape
        self.unique_classes = unique_classes
        self.noise_dim = noise_dim

        self.label_emb = nn.Embedding(self.unique_classes, self.unique_classes)

        self.encoder = nn.Sequential(
            spectral_norm(nn.Linear(self.smiles_nodes + self.unique_classes, MIN_DIM)),
            nn.BatchNorm1d(MIN_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(MIN_DIM, MIN_DIM//2)),
            nn.BatchNorm1d(MIN_DIM//2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.rnn1 = nn.LSTM(MIN_DIM//2, MIN_DIM, bidirectional=True ,batch_first=True)

        self.decoder = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM*2 + self.unique_classes + self.noise_dim, MIN_DIM*2, kernel_size=1)),
            nn.BatchNorm1d(MIN_DIM*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )

        self.rnn2 = nn.LSTM(MIN_DIM*2, MIN_DIM, bidirectional=True, batch_first=True)

        self.out = nn.Sequential(
            spectral_norm(nn.Linear(MIN_DIM*2, self.smiles_nodes)),
            nn.Tanh()
        )

    def forward(self, z, c, rs):
        z = z.view(z.size(0), self.noise_dim)
        rs = rs.view(rs.size(0), self.smiles_nodes)

        assert torch.all(c >= 0)
        assert torch.all(c < self.label_emb.num_embeddings)

        c = self.label_emb(c)

        x = torch.cat([rs.squeeze(1), c], 1)

        encoder_out = self.encoder(x)

        lstm1_out, _ = self.rnn1(encoder_out.unsqueeze(1))

        y = torch.cat([lstm1_out.squeeze(1), c, z], 1)
        
        decoder_out = self.decoder(y.unsqueeze(-1))

        lstm2_out, _ = self.rnn2(decoder_out.unsqueeze(1))
    
        out = self.out(lstm2_out)
        
        return out.squeeze()

class Discriminator(nn.Module):
    def __init__(self, smiles_nodes, smiles_shape, classes_shape, unique_classes, MIN_DIM):
        super().__init__()
        self.smiles_nodes = smiles_nodes
        self.smiles_shape = smiles_shape
        
        self.unique_classes = unique_classes
        self.classes_shape = classes_shape
        self.unique_classes = unique_classes

        self.label_emb = nn.Embedding(self.unique_classes, self.unique_classes)

        self.smile_input = nn.Sequential(
            spectral_norm(nn.Conv1d(self.smiles_nodes + self.unique_classes, MIN_DIM, kernel_size=1)),
            nn.BatchNorm1d(MIN_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        self.parallel1_ = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM, MIN_DIM//2, kernel_size=1)),
            nn.BatchNorm1d(MIN_DIM//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )

        self.parallel_1 = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM, MIN_DIM//4, kernel_size=1)),
            nn.BatchNorm1d(MIN_DIM//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            spectral_norm(nn.Linear(MIN_DIM//4, 1))
        )

        self.parallel2_ = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM//2, MIN_DIM//4, kernel_size=1)),
            nn.BatchNorm1d(MIN_DIM//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten()
        )

        self.parallel_2 = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM//2, MIN_DIM//4, kernel_size=1)),
            nn.BatchNorm1d(MIN_DIM//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            spectral_norm(nn.Linear(MIN_DIM//4, 1))
        )

        self.parallel_3 = nn.Sequential(
            spectral_norm(nn.Linear(MIN_DIM//4, 1))
        )

        self.final = nn.Sequential(
            spectral_norm(nn.Linear(3, 1))
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        x = x.view(x.size(0), self.smiles_nodes)
        c = self.label_emb(c)

        x = torch.cat([x, c], 1).unsqueeze(-1)

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
    
def check_repeated(data_list):
    unique_list = set(data_list)    

    return 1-len(unique_list)/len(data_list)

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, labels, num_classes, real_smiles):
    g_optimizer.zero_grad()
    
    z = Variable(torch.normal(mean=0, std=1, size=(batch_size, NOISE_DIM))).to(device)

    fake_labels = Variable(torch.randint(0, num_classes, size=labels.shape).to(torch.int)).to(device)

    fake_smiles = generator(z, fake_labels, real_smiles)
    validity = discriminator(fake_smiles, fake_labels)

    translated_smiles = translate_smiles(fake_smiles, dataset)

    g_smiles_validity_loss = np.mean([float(1) if Chem.MolFromSmiles(this) == None else float(0) for this in translated_smiles])
    untranslatable_loss = math.log(1 + g_smiles_validity_loss) / math.log(2)

    # duplicated_loss = backup_and_check_percentage(translated_smiles)
    g_repeated_loss = math.log(1 + check_repeated(translated_smiles)) / math.log(2)

    os.system('clear')

    # g_smiles_validity_loss = criterion(Variable(smiles_validity).to(torch.float).to(device).float(),
    #                                     Variable(torch.ones(batch_size).to(torch.float)).to(device).float())
    
    g_discriminator_loss = criterion(validity, Variable(torch.ones(batch_size).to(torch.float)).to(device).float())
    
    g_loss =  g_discriminator_loss*(1 + 2*untranslatable_loss + g_repeated_loss/2)/3.5
    
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item(), g_repeated_loss, g_discriminator_loss.data.item(), untranslatable_loss

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_smiles, labels, num_classes):
    d_optimizer.zero_grad()

    # train with real smiles

    real_validity = discriminator(real_smiles, labels)
    # real_validity = torch.where(real_validity > 0.9, torch.tensor(0.9), real_validity)

    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

    # train with fake smiles
    z = Variable(torch.normal(mean=0, std=1, size=(batch_size, NOISE_DIM))).to(device)
    
    fake_labels = Variable(torch.randint(0, num_classes, size=labels.shape).to(torch.int)).to(device)

    fake_smiles = generator(z, fake_labels, real_smiles)

    fake_validity = discriminator(fake_smiles, fake_labels)

    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item(), real_loss.data.item(),  fake_loss.data.item()

def translate_smiles(sample_smiles, dataset):
    all_gen_smiles = []
    for sml in sample_smiles:
        this_smile = np.reshape(sml.detach().cpu().numpy(), (1, dataset.smiles.shape[1], dataset.smiles.shape[2]))
        all_gen_smiles.append(dataset.coder.inverse_transform(this_smile)[0])
    return all_gen_smiles

def save_state(generator, discriminator, g_optimizer, d_optimizer,
               epoch, dataset, train_tracking, save_model_in):

    if not os.path.exists('generated_files'):
        os.mkdir('generated_files')

    folder_name = f"./generated_files/{MIN_DIM}lr{LR}g{GLRM}_bpe{BPE}"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    if epoch % save_model_in == 0:    

        for batch in generator_loader:
            sample_smiles, sample_classes = batch
            break

        sample_smiles = Variable(sample_smiles.to(torch.float)).to(device).squeeze(1)
        sample_classes = Variable(sample_classes.to(torch.int)).to(device)

        z = Variable(torch.normal(mean=0, std=1, size=(32, NOISE_DIM))).to(device)
        sample_smiles = generator(z, sample_classes, sample_smiles)
        # Translating smiles

        processed_molecules = translate_smiles(sample_smiles, dataset)
        
        for i in ("smiles", "classes"):
            if not os.path.exists(f"{folder_name}/{i}"):
                os.mkdir(f"{folder_name}/{i}")

        # Translating classes
        all_gen_classes = []
        for cls in sample_classes:
            all_gen_classes.append(dataset.classes_code[cls])
        # Saving Generator State
        generator_state = {'state_dict': generator.state_dict(), 'optimizer': g_optimizer.state_dict()}
        torch.save(generator_state, f"{folder_name}/generator.pt")

        # Saving Discriminator State
        discriminator_state = {'state_dict': discriminator.state_dict(), 'optimizer': d_optimizer.state_dict()}
        torch.save(discriminator_state, f"{folder_name}/discriminator.pt")

        with open(f"{folder_name}/smiles/cgan_{epoch}.txt", 'w') as f:
            for molecule in processed_molecules:
                f.write(molecule)
                f.write('\n')
            f.close()

        with open(f"{folder_name}/classes/cgan_{epoch}.txt", 'w') as f:
            for classes in all_gen_classes:
                f.write('@'.join(classes))
                f.write('\n')
            f.close()

    with open(f"{folder_name}/statistics.json", 'w') as f:
        json.dump(train_tracking, f)
        f.close()

    plot(f"{folder_name}/statistics.json", path_to_save=folder_name)


def train(params, generator, discriminator, criterion, batch_size=None, num_epochs = 1000, display_step = 10, num_classes = 150):

    MIN_DIM = params["min_dim"]
    BPE = params["batch_per_epoca"]
    LR = params["learning_rate"]
    GLRM = params["generator_lr_multiplier"]

    if os.path.isfile(f"./generated_files/{MIN_DIM}lr{LR}g{GLRM}_bpe{BPE}/statistics.json"):
        with open(f"./generated_files/{MIN_DIM}lr{LR}g{GLRM}_bpe{BPE}/statistics.json", 'r') as file:
            train_tracking = json.load(file)
    else:
        train_tracking = {}

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=GLRM*LR)
    
    generator, g_optimizer, discriminator, d_optimizer, start_epoch = load_states(generator, g_optimizer, discriminator, d_optimizer, params)        

    for epoch in range(start_epoch, num_epochs+1):
        this_epock_tracking = {"D Loss":[], "D Real Loss":[], "D Fake Loss":[], "G Loss":[],
                                 "G Repeatition Loss":[], "G Untranslatable Loss":[], "G Disc Loss":[]}
        
        for i, (smiles, labels) in enumerate(data_loader):

            if len(smiles) != batch_size or len(labels) != batch_size:
                continue

            real_smiles = Variable(smiles.to(torch.float)).to(device).squeeze(1)
            labels = Variable(labels.to(torch.int)).to(device)

            assert (labels<num_classes).all(), "target: {} invalid".format(labels)

            generator.train()

            if batch_size == None:
                batch_size = real_smiles.size(0)
                

            d_loss, d_real_loss, d_fake_loss = discriminator_train_step(batch_size, discriminator,
                                                generator, d_optimizer, criterion, real_smiles,
                                                labels, num_classes)

            g_loss, g_rep_loss, g_disc_loss, g_untranslatable_loss  = generator_train_step(batch_size, discriminator, generator, g_optimizer,
                                           criterion, labels, num_classes, real_smiles)
            
            this_epock_tracking["D Loss"].append(d_loss)
            this_epock_tracking["D Real Loss"].append(d_real_loss)
            this_epock_tracking["D Fake Loss"].append(d_fake_loss)
            this_epock_tracking["G Loss"].append(g_loss)
            this_epock_tracking["G Untranslatable Loss"].append(g_untranslatable_loss)
            this_epock_tracking["G Repeatition Loss"].append(g_rep_loss)
            this_epock_tracking["G Disc Loss"].append(g_disc_loss)
            print('Training model >> Epoch: [{}/{}] -- Batch: [{}]\nd_loss: {:.2f}                          |  g_loss: {:.2f}\n\
d_real_loss: {:.2f}, d_fake_loss: {:.2f}  |  g_disc_loss: {:.2f}, g_untranslatable_loss: {:.2f}, g_rep_loss: {:.2f}'.format(
                        epoch, num_epochs, i, d_loss, g_loss, d_real_loss, d_fake_loss, g_disc_loss, g_untranslatable_loss, g_rep_loss))

        generator.eval()

        train_tracking[epoch] = {"D Loss":np.mean(this_epock_tracking["D Loss"]),
                                "D Real Loss":np.mean(this_epock_tracking["D Real Loss"]),
                                "D Fake Loss":np.mean(this_epock_tracking["D Fake Loss"]),
                                "G Loss":np.mean(this_epock_tracking["G Loss"]),
                                "G Repeatition Loss":np.mean(this_epock_tracking["G Repeatition Loss"]),
                                "G Untranslatable Loss":np.mean(this_epock_tracking["G Untranslatable Loss"]),
                                "G Disc Loss":np.mean(this_epock_tracking["G Disc Loss"])
                                }
                    
        save_state(generator, discriminator, g_optimizer, d_optimizer,
                    epoch, dataset, train_tracking, display_step)

    return train_tracking


def load_states(generator, g_optimizer, discriminator, d_optimizer, this_params):
    def extract_single_integer_from_string(input_string):
        matches = re.findall(r'\d+', input_string)
        return int(matches[0])
    
    MIN_DIM = this_params["min_dim"]
    BPE = this_params["batch_per_epoca"]
    LR = this_params["learning_rate"]
    GLRM = this_params["generator_lr_multiplier"]

    folder_name = f"./generated_files/{MIN_DIM}lr{LR}g{GLRM}_bpe{BPE}"

    start_epoch = 0

    g_file_name = f"{folder_name}/generator.pt"
    d_file_name=f"{folder_name}/discriminator.pt"

    # Loading Generator State
    if os.path.isfile(g_file_name):
        print("=> loading generator checkpoint '{}'".format(g_file_name))
        checkpoint = torch.load(g_file_name)
        generator.load_state_dict(checkpoint['state_dict'])
        g_optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded generator checkpoint '{}'".format(g_file_name))

    # Loading Discriminator State
    if os.path.isfile(d_file_name):
        print("=> loading discriminator checkpoint '{}'".format(d_file_name))
        checkpoint = torch.load(d_file_name)
        discriminator.load_state_dict(checkpoint['state_dict'])
        d_optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded discriminator checkpoint '{}'".format(d_file_name))

    # Loading Final Epoch
    if os.path.isdir(f"{folder_name}/smiles"):
        files_names = os.listdir(f"{folder_name}/smiles")
        start_epoch = max([extract_single_integer_from_string(fn) for fn in files_names]) + 1
    print("=> loaded checkpoint from epoch '{}'".format(start_epoch))

    return generator, g_optimizer, discriminator, d_optimizer, start_epoch

if __name__ == "__main__":
    NOISE_DIM = 100

    dataset = DrugLikeMolecules(file_path='chebi_selected_smiles.txt'
                                )

    params = {"learning_rate": [0.0001],
              "generator_lr_multiplier": [10, 5],
              "batch_per_epoca": [256, 128],
              "min_dim":[128, 64]}
    
    params_combinations = list(product(*params.values()))

    generator_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for pc in params_combinations:    
        this_params = dict(zip(params.keys(), pc))
        print(this_params)

        MIN_DIM = this_params["min_dim"]
        BPE = this_params["batch_per_epoca"]
        LR = this_params["learning_rate"]
        GLRM = this_params["generator_lr_multiplier"]

        criterion = nn.BCELoss()
        generator = Generator(dataset.smiles_nodes, dataset.smiles.shape, dataset.classes.shape, dataset.unique_classes, NOISE_DIM, MIN_DIM).to(device)
        discriminator = Discriminator(dataset.smiles_nodes, dataset.smiles.shape, dataset.classes.shape, dataset.unique_classes, MIN_DIM).to(device)

        data_loader = torch.utils.data.DataLoader(dataset, BPE, shuffle=True)
        train(this_params, generator, discriminator, criterion, batch_size=BPE, num_epochs=500, num_classes=dataset.unique_classes)    