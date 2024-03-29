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

from rdkit.rdBase import BlockLogs

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

        if os.path.exists(f"{VOCAB_OPT}_{TOKENIZER}_smiles_vocab.npz"):
            self.load(f"{VOCAB_OPT}_{TOKENIZER}_smiles_vocab.npz")

    def fit(self, smiles_data, max_length = 150):
        for i in tqdm(range(len(smiles_data))):
            if TOKENIZER == "ATOM":
                smiles_data[i] = smiles_data[i].ljust(max_length)
            else:
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
            if TOKENIZER == "ATOM":
                smiles_data[i] = smiles_data[i].ljust(self.max_length)
            else:
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

replace_dict = {'Ag':'D', 'Al':'E', 'Ar':'G', 'As':'J', 'Au':'Q', 'Ba':'X', 'Be':'Y',
                'Br':'f', 'Ca':'h', 'Cd':'j', 'Ce':'k', 'Cl':'m', 'Cn':'p', 'Co':'q',
                'Cr':'v', 'Cu':'w', 'Fe':'x', 'Hg':'y', 'Ir':'z', 'La':'!', 'Mg':'$',
                'Mn':'¬', 'Mo':'&', 'Na':'_', 'Ni':'£', 'Pb':'¢', 'Pt':'?', 'Ra':'ª',
                'Ru':'º', 'Sb':';', 'Sc':':', 'Se':'>', 'Si':'<', 'Sn':'§', 'Sr':'~',
                'Te':'^', 'Tl':'|', 'Zn':'{', '@@':'}'}

inverse_dict = {v: k for k, v in replace_dict.items()}

def translate_smiles(sample_smiles, dataset):
    all_gen_smiles = []
    for sml in sample_smiles:
        this_smile = np.reshape(sml.detach().cpu().numpy(), (1, dataset.smiles.shape[1], dataset.smiles.shape[2]))
        all_gen_smiles.append(dataset.coder.inverse_transform(this_smile)[0])
    processed_molecules = preprocessing_data(all_gen_smiles, inverse_dict)
    return [gen_smiles for gen_smiles in processed_molecules[0]]

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

def preprocessing_data(molecules, replacement=None, saving=False):

    if replacement == None:
        molecules = [smi_tokenizer(mol) for mol in molecules]

    if not saving:
        molecules = pd.Series(molecules)
    else:
        molecules = pd.Series([mol for mol in molecules])

    if replacement != None:
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
    molecules = []

    for s in smiles:
        for c in s[1].split("@"):
            molecules.append(s[0])
            classes.append(c)

    classes_hot_array = np.array([])

    if not os.path.exists(f'{VOCAB_OPT}_classes_vocab.json'):
        unique_elements = list(set(classes))

        for classe in classes:
            classes_hot_array = np.append(classes_hot_array, unique_elements.index(classe))

        classes_vocab = {}
        for i, elem in enumerate(unique_elements):
            classes_vocab[elem] = i

        with open(f'{VOCAB_OPT}_classes_vocab.json', 'w') as f:
            json.dump(classes_vocab, f)
            f.close()
    else:
        with open(f'{VOCAB_OPT}_classes_vocab.json', 'r') as f:
            classes_vocab = json.load(f)
            f.close()

        for classe in classes:
            classes_hot_array = np.append(classes_hot_array, classes_vocab[classe])

        unique_elements = list(classes_vocab.keys())

    if TOKENIZER == "ATOM":
        processed_molecules, smiles_max_length = preprocessing_data(molecules, replace_dict)
    else:
        processed_molecules, smiles_max_length = preprocessing_data(molecules)

    ######## TURNING TO ONE HOT ########
    coder = smiles_coder()
    if not os.path.exists(f'{VOCAB_OPT}_{TOKENIZER}_smiles_vocab.npz'):
        coder.fit(processed_molecules, smiles_max_length)
        coder.save(f'{VOCAB_OPT}_{TOKENIZER}_smiles_vocab.npz')
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
    def __init__(self, smiles_nodes, smiles_shape, classes_shape, unique_classes, latent_dim, MIN_DIM):
        super().__init__()
        self.smiles_nodes = smiles_nodes
        self.smiles_shape = smiles_shape

        self.classes_shape = classes_shape
        self.unique_classes = unique_classes
        self.latent_dim = latent_dim

        self.label_emb = nn.Sequential(
            nn.Embedding(self.unique_classes, MIN_DIM),
            nn.Linear(MIN_DIM, self.smiles_nodes)
        )

        self.encoder = nn.Sequential(
            nn.Linear(2*self.smiles_nodes + self.latent_dim, MIN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(MIN_DIM, self.latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        self.mean_layer = nn.Linear(self.latent_dim, 2)
        self.logvar_layer = nn.Linear(self.latent_dim, 2)
        
        self.decoder = nn.Sequential(
            nn.Linear(2, self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, MIN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(MIN_DIM, self.smiles_nodes),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, c, x, z):
        x = x.view(x.size(0), self.smiles_nodes)
        c = self.label_emb(c)

        x = torch.cat([x.squeeze(1), c, z.squeeze(1)], 1)

        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        self.parallel1_ = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM, MIN_DIM//2, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )

        self.parallel_1 = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM, MIN_DIM//4, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            spectral_norm(nn.Linear(MIN_DIM//4, 1))
        )

        self.parallel2_ = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM//2, MIN_DIM//4, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten()
        )

        self.parallel_2 = nn.Sequential(
            spectral_norm(nn.Conv1d(MIN_DIM//2, MIN_DIM//4, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            spectral_norm(nn.Linear(MIN_DIM//4, 1))
        )

        self.parallel_3 = nn.Sequential(
            spectral_norm(nn.Linear(MIN_DIM//4, 1))
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
        
        out = self.sigmoid(x_concatenated)
    
        return out.squeeze()
    
def check_repeated(data_list):
    unique_list = set(data_list)    

    return (1-len(unique_list)/len(data_list))*100

def generator_loss(validity, criterion):
    total_loss = 0.0

    for guess in validity:
        modulating_factor = (1 - guess)**2
        loss = modulating_factor * criterion(guess, torch.ones_like(guess).to(device))
        total_loss += loss

    return total_loss / len(validity)

def vae_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat.reshape(x.shape), x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return (reproduction_loss + KLD) / len(x)

# Discriminator Loss Function
def discriminator_loss(real_validity, fake_validity, criterion):
    total_real_loss = 0.0
    total_fake_loss = 0.0

    for (r_guess, f_guess) in zip(real_validity, fake_validity):
        modulating_factor_real = (1 - r_guess)**2
        modulating_factor_fake = (1 - f_guess)**2

        loss_real = modulating_factor_real * criterion(r_guess, torch.ones_like(r_guess).to(device))
        loss_fake = modulating_factor_fake * criterion(f_guess, torch.zeros_like(f_guess).to(device))
        
        total_real_loss += loss_real
        total_fake_loss += loss_fake

    return total_real_loss / len(real_validity), total_fake_loss / len(fake_validity)

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, labels, num_classes, real_smiles):
    g_optimizer.zero_grad()
    
    fake_labels = Variable(torch.randint(0, num_classes, size=labels.shape).to(torch.int)).to(device)
    z = Variable(torch.normal(mean=0, std=1, size=(batch_size, LATENT_DIM))).to(device)

    fake_smiles, mean, logvar = generator(fake_labels, real_smiles, z)

    vae_loss = vae_loss_function(real_smiles, fake_smiles, mean, logvar)

    validity = discriminator(fake_smiles, fake_labels)

    translated_smiles = translate_smiles(fake_smiles, dataset)

    block = BlockLogs()

    untranslatable_loss = torch.nn.functional.binary_cross_entropy(torch.tensor([float(1) if Chem.MolFromSmiles(this) == None else float(0) for this in translated_smiles], dtype=torch.float32), torch.zeros(batch_size))

    del block
        
    g_repeated_loss = torch.tensor(check_repeated(translated_smiles))
    
    g_discriminator_loss = torch.sum(generator_loss(validity, criterion))
    
    g_loss =  (1 + g_discriminator_loss)*(untranslatable_loss + g_repeated_loss + vae_loss)
    
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item(), g_repeated_loss.data.item(), g_discriminator_loss.data.item(), untranslatable_loss.data.item(), vae_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_smiles, labels, num_classes):
    d_optimizer.zero_grad()

    # train with real smiles
    real_validity = discriminator(real_smiles, labels)

    # train with fake smiles
    fake_labels = Variable(torch.randint(0, num_classes, size=labels.shape).to(torch.int)).to(device)
    z = Variable(torch.normal(mean=0, std=1, size=(batch_size, LATENT_DIM))).to(device)

    fake_smiles, _, _ = generator(fake_labels, real_smiles, z)
    fake_validity = discriminator(fake_smiles, fake_labels)

    d_real_loss, d_fake_loss = discriminator_loss(real_validity, fake_validity, criterion)
    d_real_loss = torch.sum(d_real_loss)
    d_fake_loss = torch.sum(d_fake_loss)

    d_loss = d_real_loss + d_fake_loss

    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item(), d_real_loss.data.item(), d_fake_loss.data.item()

def save_state(generator, discriminator, g_optimizer, d_optimizer,
               epoch, dataset, train_tracking, save_model_in,
               force_break = False, force_save = False):

    print('Saving state...')
    if epoch % save_model_in == 0 and force_break == False or force_save == True:   

        for batch in generator_loader:
            sample_smiles, sample_classes = batch
            break

        sample_smiles = Variable(sample_smiles.to(torch.float)).to(device).squeeze(1)
        sample_classes = Variable(sample_classes.to(torch.int)).to(device)
        z = Variable(torch.normal(mean=0, std=1, size=(32, LATENT_DIM))).to(device)

        sample_smiles, _, _ = generator(sample_classes, sample_smiles, z)
        # Translating smiles

        processed_molecules = translate_smiles(sample_smiles, dataset)
        
        for i in ("smiles", "classes"):
            if not os.path.exists(f"{FOLDER_PATH}/{i}"):
                os.mkdir(f"{FOLDER_PATH}/{i}")

        # Translating classes
        all_gen_classes = []
        for cls in sample_classes:
            all_gen_classes.append(dataset.classes_code[cls])
        # Saving Generator State
        generator_state = {'state_dict': generator.state_dict(), 'optimizer': g_optimizer.state_dict()}
        torch.save(generator_state, f"{FOLDER_PATH}/generator.pt")

        # Saving Discriminator State
        discriminator_state = {'state_dict': discriminator.state_dict(), 'optimizer': d_optimizer.state_dict()}
        torch.save(discriminator_state, f"{FOLDER_PATH}/discriminator.pt")

        with open(f"{FOLDER_PATH}/smiles/cgan_{epoch}.txt", 'w') as f:
            for molecule in processed_molecules:
                f.write(molecule)
                f.write('\n')
            f.close()

        with open(f"{FOLDER_PATH}/classes/cgan_{epoch}.txt", 'w') as f:
            for classes in all_gen_classes:
                f.write('@' + classes + '\n')
            f.close()

    with open(f"{FOLDER_PATH}/statistics.json", 'w') as f:
        json.dump(train_tracking, f)
        f.close()

    plot(f"{FOLDER_PATH}/statistics.json", path_to_save=FOLDER_PATH, file_name=f"{TOKENIZER}_{VOCAB_OPT}_{MIN_DIM}_{GEN_OPT_STR}_lr{LR}_gen{GLRM}_bs{BPE}")

def load_states(generator, g_optimizer, discriminator, d_optimizer):
    def extract_single_integer_from_string(input_string):
        matches = re.findall(r'\d+', input_string)
        return int(matches[0])

    start_epoch = 0

    g_file_name = f"{FOLDER_PATH}/generator.pt"
    d_file_name=f"{FOLDER_PATH}/discriminator.pt"

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
    if os.path.isdir(f"{FOLDER_PATH}/smiles"):
        files_names = os.listdir(f"{FOLDER_PATH}/smiles")
        start_epoch = max([extract_single_integer_from_string(fn) for fn in files_names]) + 1
    print("=> loaded checkpoint from epoch '{}'".format(start_epoch))

    return generator, g_optimizer, discriminator, d_optimizer, start_epoch

def train(generator, discriminator, criterion, batch_size=None, num_epochs = 1000, display_step = 10, num_classes = 150):

    best_validation_loss = float('inf')
    current_patience = 0
    force_break = False

    if os.path.isfile(f"{FOLDER_PATH}/statistics.json"):
        with open(f"{FOLDER_PATH}/statistics.json", 'r') as file:
            train_tracking = json.load(file)
    else:
        train_tracking = {}

    d_optimizer = GEN_OPT(discriminator.parameters(), lr=LR, weight_decay=0.001)
    g_optimizer = GEN_OPT(generator.parameters(), lr=GLRM*LR, weight_decay=0.001)
    
    generator, g_optimizer, discriminator, d_optimizer, start_epoch = load_states(generator, g_optimizer, discriminator, d_optimizer)        

    for epoch in range(start_epoch, num_epochs+1):
        this_epock_tracking = {"D Loss":[], "D Real Loss":[], "D Fake Loss":[], "G Loss":[],
                                 "G Untranslatable Loss":[], "G Disc Loss":[], "G Repetition Loss":[],
                                 "G VAE Loss":[]}
        
        for i, (smiles, labels) in enumerate(data_loader):

            if len(smiles) != batch_size or len(labels) != batch_size:
                continue

            real_smiles = Variable(smiles.to(torch.float)).to(device).squeeze(1)
            labels = Variable(labels.to(torch.int)).to(device)

            assert (labels<num_classes).all(), "target: {} invalid".format(labels)

            generator.train()

            if batch_size == None:
                batch_size = real_smiles.size(0)
                

            d_loss, d_real, d_fake = discriminator_train_step(batch_size, discriminator,
                                                generator, d_optimizer, criterion, real_smiles,
                                                labels, num_classes)

            g_loss, g_rep_loss, g_disc_loss, g_untranslatable_loss, g_vae_loss  = generator_train_step(batch_size, discriminator, generator, g_optimizer,
                                           criterion, labels, num_classes, real_smiles)
            
            this_epock_tracking["D Loss"].append(d_loss)
            this_epock_tracking["D Real Loss"].append(d_real)
            this_epock_tracking["D Fake Loss"].append(d_fake)
            this_epock_tracking["G Loss"].append(g_loss)
            this_epock_tracking["G Repetition Loss"].append(g_rep_loss)
            this_epock_tracking["G Disc Loss"].append(g_disc_loss)
            this_epock_tracking["G Untranslatable Loss"].append(g_untranslatable_loss)
            this_epock_tracking["G VAE Loss"].append(g_vae_loss)

            
            if i%5==0:
                os.system('clear')

                print('Path:', FOLDER_PATH)
                print('Training model >> Epoch: [{}/{}] -- Batch: [{}]\nd_loss: {:.7f}  |  g_loss: {:.7f}\n\
                |  g_disc_loss: {:.7f}, g_untranslatable_loss: {:.7f}, g_rep_loss: {:.7f}, g_vae_loss {:.7f}'.format(
                            epoch, num_epochs, i, d_loss, g_loss, g_disc_loss, g_untranslatable_loss, g_rep_loss, g_vae_loss))

                print('Best val:', best_validation_loss)
                print('Current patience:', current_patience)

        generator.eval()

        train_tracking[epoch] = {"D Loss":np.mean(this_epock_tracking["D Loss"]),
                                 "D Real Loss":np.mean(this_epock_tracking["D Real Loss"]),
                                 "D Fake Loss":np.mean(this_epock_tracking["D Fake Loss"]),
                                "G Loss":np.mean(this_epock_tracking["G Loss"]),
                                "G Untranslatable Loss":np.mean(this_epock_tracking["G Untranslatable Loss"]),
                                "G Disc Loss":np.mean(this_epock_tracking["G Disc Loss"]),
                                "G Repetition Loss":np.mean(this_epock_tracking["G Repetition Loss"]),
                                "G VAE Loss":np.mean(this_epock_tracking["G VAE Loss"])
                                }


        if np.mean(this_epock_tracking["G Loss"]) < best_validation_loss:
            best_validation_loss = np.mean(this_epock_tracking["G Loss"])
            current_patience = 0

            if epoch >= 50:
                save_state(generator, discriminator, g_optimizer, d_optimizer,
                        epoch, dataset, train_tracking, display_step,
                        force_break, True)
        else:
            current_patience += 1

        if current_patience >= patience:
            force_break = True


        save_state(generator, discriminator, g_optimizer, d_optimizer,
                    epoch, dataset, train_tracking, display_step,
                    force_break
                    )


        if force_break:
            return train_tracking
        
    return train_tracking

if __name__ == "__main__":
    patience = 5

    params = {
              "tokenizer":["ATOM", "FRAGMENT"],
              "latent_dim":[64],
              "batch_per_epoca": [64],
              "generator_lr_multiplier": [5],
              "min_dim":[128],
              "learning_rate": [0.00001],
              "gen_opt":[
                            torch.optim.RMSprop,
                            torch.optim.Adamax,
                            torch.optim.SGD,
                            torch.optim.Adam,
                         ],
              "dataset":['chebi_selected_smiles.txt', 'dense_chebi_selected_smiles.txt']
            }
    
    params_combinations = list(product(*params.values()))
    
    for pc in params_combinations:    
        selected_params = dict(zip(params.keys(), pc))
        print(selected_params)

        TOKENIZER = selected_params["tokenizer"]
        LATENT_DIM = selected_params["latent_dim"]
        MIN_DIM = selected_params["min_dim"]
        BPE = selected_params["batch_per_epoca"]
        LR = selected_params["learning_rate"]
        GLRM = selected_params["generator_lr_multiplier"]
        DATASET = selected_params["dataset"]
        VOCAB_OPT = "DENSE" if "dense_" in DATASET else "SPARSE"
        GEN_OPT = selected_params["gen_opt"]
        GEN_OPT_STR = "ADAM" if GEN_OPT == torch.optim.Adam else ("SGD" if GEN_OPT == torch.optim.SGD else ("ADAMAX" if GEN_OPT == torch.optim.Adamax else "RMS"))

        FOLDER_PATH = f"n_generated_files/{TOKENIZER}/{VOCAB_OPT}/{MIN_DIM}/{GEN_OPT_STR}/lr{LR}_gen{GLRM}_bs{BPE}"

        os.makedirs(FOLDER_PATH, exist_ok=True)

        dataset = DrugLikeMolecules(file_path=DATASET)

        generator_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.BCELoss()
        generator = Generator(dataset.smiles_nodes, dataset.smiles.shape, dataset.classes.shape, dataset.unique_classes, LATENT_DIM, MIN_DIM).to(device)
        discriminator = Discriminator(dataset.smiles_nodes, dataset.smiles.shape, dataset.classes.shape, dataset.unique_classes, MIN_DIM).to(device)

        data_loader = torch.utils.data.DataLoader(dataset, BPE, shuffle=True)
        train(generator, discriminator, criterion, batch_size=BPE, num_epochs=300, num_classes=dataset.unique_classes)    
