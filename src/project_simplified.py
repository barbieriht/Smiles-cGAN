# Bibliotecas
from numpy import asarray
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense, Reshape, Embedding, Concatenate
# from keras_nlp.layers import TransformerEncoder
from keras.layers import BatchNormalization, LeakyReLU, Flatten
from keras.optimizers import Adam
from visualize_statistics import plot
import json
import tensorflow as tf
from itertools import product

import numpy as np
import os
import pandas as pd
from smiles import smiles_coder

def constroi_gerador(learning_rate):
    print("=== CONSTROI GERADOR ===")
    n_nodes = smile_nodes + classes_nodes

    in_label = Input(shape=(1,), name="in_label") 
    li = Embedding(classes_nodes, 50)(in_label)
    li = Dense(n_nodes, name="gen_dense1")(li)
    li = Reshape((n_nodes,))(li)

    in_lat = Input(shape=(latent_dim,), name="in_lat")

    gen = Dense(n_nodes, name="gen_dense2")(in_lat)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU()(gen)

    merge = Concatenate()([gen, li])
    gen = Dense(256, name="gen_dense3")(merge)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU()(gen)

    gen = Dense(256, name="gen_dense4")(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU()(gen)

    smile = Dense(n_nodes, activation='tanh', name="gen_dense5")(gen)
    smile = Flatten()(smile)

    model = Model([in_lat,in_label], outputs=smile, name="Gerador")

    model.summary()

    return model

# Discriminador -> classificador binário.
# Pega uma imagem e classifica como verdadeira ou falsa.
def constroi_discriminador(learning_rate):
    print("=== CONSTROI DISCRIMINADOR ===")
    n_nodes = smile_nodes + classes_nodes

    disc_input = Input(shape=(n_nodes,), name="smile_input")

    disc = Dense(n_nodes, name="disc_dense1")(disc_input)
    disc = BatchNormalization()(disc)
    disc = LeakyReLU()(disc)

    disc = Dense(256, name="disc_dense2")(disc)
    disc = BatchNormalization()(disc)
    disc = LeakyReLU()(disc)

    disc = Dense(128, name="disc_dense3")(disc)
    disc = BatchNormalization()(disc)
    disc = LeakyReLU()(disc)

    disc = Dropout(0.4)(disc)

    out_layer = Dense(1, activation="sigmoid", name='output_layer')(disc)

    model = Model(inputs=disc_input, outputs=out_layer, name="Discriminador")

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    model.summary()

    return model

# Define a GAN
def define_gan(g_model,d_model, learning_rate):
    print("=== DEFINE GAN ===")
    z = Input(shape=(latent_dim,))
    label = Input(shape=(1,))

    smile = g_model([z, label])
    smile = Flatten()(smile)

    print("Smile shape:", smile.shape)
    print("Label shape:", label.shape)

    d_model.trainable = False

    prediction = d_model(smile)

    print("Prediction shape:", prediction.shape)

    model = Model([z, label], prediction)

    opt = Adam(learning_rate=learning_rate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def gera_amostras_reais(dataset,n_batch):

    # print("=== GERA AMOSTRAS REAIS ===")
    smiles, labels = dataset
    # print("Smiles shape:", smiles.shape)
    # print("Labels shape:", labels.shape)

    idx = randint(0,smiles.shape[0], n_batch)
    # print("Idx shape:", idx.shape)
    # Select random images and labels
    X, labels = smiles[idx], labels[idx]
    # print(f"X: {X.shape}")
    X = np.reshape(X, (n_batch, smile_nodes))

    # print(f"X: {X.shape}")
    # print(f"labels: {labels.shape}")
    # Call those images real
    y = ones((n_batch,1))
    # print("y shape:", y.shape)
    return [X,labels],y

def gera_amostras_falsas(gerador,latent_dim,n_batch):
    # print("=== GERA AMOSTRAS FALSAS ===")
    # Gera ruido
    z_input, label_input = gera_ruido(latent_dim,n_batch,classes_nodes)
    # print(f"label_input: {label_input.shape}")

    # Gera as imagens
    images = gerador.predict([z_input, label_input])
    # print(f"images: {images.shape}")

    # Cria a classe
    y = zeros((n_batch,1))

    return images,y

# Gera vetores latentes (ruido)
def gera_ruido(latent_dim,n_batch,n_classes):
    # print("=== GERA RUÍDO ===")
    # Ruído: um array aleatório de tamanho = 100 * 64
    x_input = randn(latent_dim * n_batch)
    # print("X input shape:", x_input.shape)
    # O Ruído é passado para o formato (64, 100)
    z_input = x_input.reshape(n_batch,latent_dim)
    # print("Z input shape:", z_input.shape)
    # Os rótulos são de formato (64, 1) / uma classe sorteada pra cada item do batch
    labels = randint(0,n_classes,n_batch)
    # print("Labels shape:", labels.shape)
    return [z_input, labels]

# Treinamento 
def treinamento(gerador,discriminador,gan,dataset,latent_dim,
                n_epocas=100, bat_per_epoca=20, n_batch=128, save_interval=10):
    print("=== TREINAMENTO ===")
    train_tracking = {}

    meio_batch = int(n_batch / 2)

    # Para cada época
    for epoca in range(n_epocas):
        # print(f"Epoca: {epoca}")
        # Para cada batch
        for batch in range(bat_per_epoca):
            # print(f"Batch: {batch}")
            # Treinamento do discriminador
            # Amostras reais
            [X_real, labels_real], y_real = gera_amostras_reais(dataset,meio_batch)

            # print("Before train on real batch...")
            # print(f"X_real: {X_real.shape}")
            # print(f"labels_real: {labels_real.shape}")
            # print(f"y_real: {y_real.shape}")

            disc_input_real = Concatenate(axis=1)([X_real, labels_real])
            disc_input_real = Flatten()(disc_input_real)

            d_loss_real, _ = discriminador.train_on_batch(disc_input_real, y_real)

            # Amostras falsas
            disc_input_fake, y_fake = gera_amostras_falsas(gerador,latent_dim,meio_batch)

            # print("Before train on fake batch...")
            # print(f"X_fake: {X_fake.shape}")
            # print(f"labels_fake: {labels_fake.shape}")
            # print(f"y_fake: {y_fake.shape}")

            d_loss_fake, _ = discriminador.train_on_batch(disc_input_fake, y_fake)

            # Treinamento da GAN
            [z_input, labels_input] = gera_ruido(latent_dim,n_batch,classes_nodes)
            y_gan = ones((n_batch,1))

            # Treina
            gan_loss = gan.train_on_batch([z_input, labels_input],y_gan)

            # Imprimir
            print('Epoca: %d, Batch: %d/%d, dR=%.3f, dF=%.3f, g=%.3f' %
                    (epoca+1, batch+1, bat_per_epoca, d_loss_real, d_loss_fake, gan_loss))
        
        if epoca % save_interval == 0:
            save_smiles(epoca)

        train_tracking[epoca+1] = {"Discriminator Real Loss":d_loss_real, "Discriminator Fake Loss":d_loss_fake, "GAN Loss":gan_loss}

        with open('train_statistics.json', 'w') as json_file:
            json.dump(train_tracking, json_file)

    # Salva o melhor modelo
    gerador.save('smiles_cgan_gen_best.h5')

# # This function saves our images for us to view
def save_smiles(epoch):
    z_input, label_input = gera_ruido(latent_dim, 64, classes_nodes)
    gen_smiles = gerador.predict([z_input, label_input])

    # Rescale images 0 - 1
    gen_smiles = 0.5 * gen_smiles + 0.5

    if not os.path.exists('./smiles'):
        os.mkdir('./smiles')
        
    all_gen_smiles = []
    for sml in gen_smiles:
        this_smile = np.reshape(sml, smiles_shape)
        all_gen_smiles.append(coder.inverse_transform(this_smile)[0])
    processed_molecules = preprocessing_data(all_gen_smiles, inverse_dict)

    with open('smiles/cgan_%d.txt' % epoch, 'a') as f:
        for molecule in processed_molecules:
            f.write(molecule)
            f.write('\n')
        f.close()

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

if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)

    replace_dict = {'Ag':'D', 'Al':'E', 'Ar':'G', 'As':'J', 'Au':'Q', 'Ba':'X', 'Be':'Y',
                    'Br':'f', 'Ca':'h', 'Cd':'j', 'Ce':'k', 'Cl':'m', 'Cn':'p', 'Co':'q',
                    'Cr':'v', 'Cu':'w', 'Fe':'x', 'Hg':'y', 'Ir':'z', 'La':'!', 'Mg':'$',
                    'Mn':'¬', 'Mo':'&', 'Na':'_', 'Ni':'£', 'Pb':'¢', 'Pt':'?', 'Ra':'ª',
                    'Ru':'º', 'Sb':';', 'Sc':':', 'Se':'>', 'Si':'<', 'Sn':'§', 'Sr':'~',
                    'Te':'^', 'Tl':'|', 'Zn':'{', '@@':'}'}

    inverse_dict = {v: k for k, v in replace_dict.items()}
    
    # smiles_arrays, training_set, classes_arrays, coder = get_hot_smiles("chebi_smiles_com_classes.txt")
    smiles_arrays, training_set, classes_arrays, coder = get_hot_smiles("chebi_smiles_1of10_subset.txt")
    smiles_shape = smiles_arrays.shape
    smiles_shape = (smiles_shape[1], smiles_shape[2], 1)

    classes_shape = (classes_arrays.shape[1],1)

    smile_nodes = smiles_shape[0] * smiles_shape[1]
    classes_nodes = classes_shape[0]

    print(f"Smiles shape: {smiles_shape}\nClasses shape: {classes_shape}\nSmiles nodes: {smile_nodes}\nClasses nodes: {classes_nodes}")
    latent_dim = 100

    params = {"learning_rate": [0.01, 0.001, 0.0001],
              "bat_per_epoca": [12, 24, 48]}
    
    combinations = list(product(*params.values()))

    for param in combinations:
        this_params = dict(zip(params.keys(), param))
        
        discriminador = constroi_discriminador(learning_rate=this_params["learning_rate"])

        gerador = constroi_gerador(learning_rate=this_params["learning_rate"])

        gan = define_gan(gerador, discriminador, this_params["learning_rate"])

        treinamento(gerador,discriminador,gan,[smiles_arrays, classes_arrays],latent_dim, bat_per_epoca=this_params["bat_per_epoca"], n_epocas=100)
    
        plot(bat_per_epoca=this_params["bat_per_epoca"], learning_rate=this_params["learning_rate"])