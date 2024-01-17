# Bibliotecas
from numpy import asarray
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Embedding, Concatenate, Dropout
from keras_nlp.layers import TransformerEncoder
from keras.layers import BatchNormalization, LeakyReLU
from keras.optimizers import Adam

import tensorflow as tf

import numpy as np
import os
import pandas as pd
from seq_encode.smiles import smiles_coder

# Gerador
def constroi_gerador():
    # print("CONSTROI GERADOR")

    model = Sequential()

    model.add(TransformerEncoder(64, 8))

    model.add(Dense(64, activation="LeakyReLU"))
    model.add(BatchNormalization())

    model.add(Dense(128, activation="LeakyReLU"))
    model.add(BatchNormalization())

    model.add(Dense(256, activation="LeakyReLU"))
    model.add(BatchNormalization())

    model.add(Dense(smile_nodes, activation="LeakyReLU"))

    model.add(TransformerEncoder(64, 8))

    in_label = Input(shape=(1,)) 
    in_lat = Input(shape=(latent_dim,))

    label_embedding = Embedding(classes_nodes, latent_dim)(in_label)
    label_embedding = Reshape((-1,))(label_embedding)

    joined_representation = Concatenate()([in_lat, label_embedding])

    smile = model(joined_representation)

    # print(f"Generator shapes:\n'inputs: {[in_lat,in_label]}\noutput: {smile}")
    model = Model([in_lat,in_label],smile)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    model.summary()

    return model

# Discriminador -> classificador binário.
# Pega uma imagem e classifica como verdadeira ou falsa.
def constroi_discriminador():
    # print("CONSTROI DISCRIMINADOR")

    in_label = Input(shape=(1,))
    smile_input = Input(shape=(smile_nodes,))

    # smile_input = Flatten()(smile_input)
    # print(f"smile_input: {smile_input.shape}")

    model = Sequential()

    transformer1 = TransformerEncoder(64, 8)(smile_input)

    big_hidden_layer1 = Dense(256, activation="LeakyReLU")(transformer1)
    big_bn1 = BatchNormalization()(big_hidden_layer1)

    big_hidden_layer2 = Dense(128, activation="LeakyReLU")(big_bn1)
    big_bn2 = BatchNormalization()(big_hidden_layer2)

    big_hidden_layer3 = Dense(64, activation="LeakyReLU")(big_bn2)
    big_bn3 = BatchNormalization()(big_hidden_layer3)

    big_hidden = Dense(1, activation="sigmoid", name='big_hidden')(big_bn3)


    medium_hidden_layer2 = Dense(128, activation="LeakyReLU")(big_hidden_layer1)
    medium_bn2 = BatchNormalization()(medium_hidden_layer2)

    medium_hidden_layer3 = Dense(64, activation="LeakyReLU")(medium_bn2)
    medium_bn3 = BatchNormalization()(medium_hidden_layer3)

    medium_hidden = Dense(1, activation="sigmoid", name='medium_hidden')(medium_bn3)


    small_hidden_layer3 = Dense(64, activation="LeakyReLU")(medium_hidden_layer2)
    small_bn3 = BatchNormalization()(small_hidden_layer3)

    small_hidden = Dense(1, activation="sigmoid", name='small_hidden')(small_bn3)

    concatenated_output = Concatenate()([big_hidden, medium_hidden, small_hidden])

    dropout = Dropout(0.4)(concatenated_output)

    transformer2 = TransformerEncoder(64, 8)(dropout)

    out_layer = Dense(1, activation="sigmoid", name='output_layer')(transformer2)

    model = Model(inputs=[smile_input, in_label], outputs=out_layer)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    model.summary()

    return model

# Define a GAN
def define_gan(g_model,d_model):
    # print("DEFINE GAN")
    
    z = Input(shape=(100,))
    label = Input(shape=(1,))

    # print("label:", label)
    smile = g_model([z, label])

    d_model.trainable = False

    prediction = d_model([smile, label])

    # print("prediction:", prediction)
    # Conditional (Conditional) GAN model with fixed discriminator to train the generator
    cgan = Model([z, label], prediction)
    cgan.compile(loss='categorical_crossentropy', optimizer=Adam())

    return cgan

# def product_loss(y_true, y_pred):
#     loss1 = binary_crossentropy(y_true[0], y_pred[0])
#     loss2 = binary_crossentropy(y_true[1], y_pred[1])
#     loss3 = binary_crossentropy(y_true[2], y_pred[2])
#     return keras.backend.prod([loss1, loss2, loss3])

# Gera amostras reais
def gera_amostras_reais(dataset,n_batch):

    # print("GERA AMOSTRAS REAIS")
    smiles, labels = dataset
    idx = randint(0,smiles.shape[0], n_batch)
    # Select random images and labels
    X, labels = smiles[idx], labels[idx]
    X = np.reshape(X, (n_batch, smile_nodes))

    # print(f"X: {X.shape}")
    # print(f"labels: {labels.shape}")
    # Call those images real
    y = ones((n_batch,1))

    return [X,labels],y

# Gerar amostras falsas
def gera_amostras_falsas(gerador,latent_dim,n_batch):
    # print("GERA AMOSTRAS FALSAS")
    # Gera ruido
    z_input, label_input = gera_ruido(latent_dim,n_batch)
    # print(f"label_input: {label_input.shape}")

    # Gera as imagens
    images = gerador.predict([z_input, label_input])
    # print(f"images: {images.shape}")

    # Cria a classe
    y = zeros((n_batch,1))

    return [images,label_input],y

# Gera vetores latentes (ruido)
def gera_ruido(latent_dim,n_batch,n_classes=10):
    # print("GERA RUÍDO")
    # ruidos
    x_input = randn(latent_dim * n_batch)

    # Reshape
    z_input = x_input.reshape(n_batch,latent_dim)

    # Gera os rotulos
    labels = randint(0,n_classes,n_batch)

    return [z_input, labels]

# Treinamento 
def treinamento(gerador,discriminador,gan,dataset,latent_dim,
                n_epocas=100,n_batch=128, save_interval=10):
    
    bat_per_epoca = 100
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
            # print("Before train on batch...")
            # print(f"X_real: {X_real.shape}")
            # print(f"labels_real: {labels_real.shape}")
            # print(f"y_real: {y_real.shape}")

            d_loss_real, _ = discriminador.train_on_batch([X_real, labels_real], y_real)

            # Amostras falsas
            [X_fake,labels_fake], y_fake = gera_amostras_falsas(gerador,latent_dim,meio_batch)

            # print("Before train on batch...")
            # print(f"X_fake: {X_fake.shape}")
            # print(f"labels_fake: {labels_fake.shape}")
            # print(f"y_fake: {y_fake.shape}")
            d_loss_fake, _ = discriminador.train_on_batch([X_fake,labels_fake], y_fake)


            # Treinamento da GAN
            [z_input, labels_input] = gera_ruido(latent_dim,n_batch)
            y_gan = ones((n_batch,1))

            # Treina
            gan_loss = gan.train_on_batch([z_input, labels_input],y_gan)

            # Imprimir
            print('Epoca: %d, Batch: %d/%d, dR=%.3f, dF=%.3f, g=%.3f' %
                    (epoca+1, batch+1, bat_per_epoca, d_loss_real, d_loss_fake, gan_loss))
            
            if epoca % save_interval == 0:
                save_smiles(epoca)
    
    # Salva o melhor modelo
    gerador.save('smiles_cgan_gen_best.h5')

# # This function saves our images for us to view
def save_smiles(epoch):
    z_input, label_input = gera_ruido(latent_dim, 64)
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
    replace_dict = {'Ag':'D', 'Al':'E', 'Ar':'G', 'As':'J', 'Au':'Q', 'Ba':'X', 'Be':'Y',
                    'Br':'f', 'Ca':'h', 'Cd':'j', 'Ce':'k', 'Cl':'m', 'Cn':'p', 'Co':'q',
                    'Cr':'v', 'Cu':'w', 'Fe':'x', 'Hg':'y', 'Ir':'z', 'La':'!', 'Mg':'$',
                    'Mn':'¬', 'Mo':'&', 'Na':'_', 'Ni':'£', 'Pb':'¢', 'Pt':'?', 'Ra':'ª',
                    'Ru':'º', 'Sb':';', 'Sc':':', 'Se':'>', 'Si':'<', 'Sn':'§', 'Sr':'~',
                    'Te':'^', 'Tl':'|', 'Zn':'{', '@@':'}'}

    inverse_dict = {v: k for k, v in replace_dict.items()}
    
    smiles_arrays, training_set, classes_arrays, coder = get_hot_smiles("chebi_smiles_com_classes.txt")
    # smiles_arrays, training_set, classes_arrays, coder = get_hot_smiles("chebi_smiles_1of10_subset.txt")

    smiles_shape = smiles_arrays.shape
    smiles_shape = (smiles_shape[1], smiles_shape[2], 1)

    print(smiles_shape)
    classes_shape = classes_arrays.shape


    smile_nodes = smiles_shape[0] * smiles_shape[1]
    classes_nodes = classes_shape[1]

    latent_dim = 100

    discriminador = constroi_discriminador()

    gerador = constroi_gerador()

    gan = define_gan(gerador,discriminador)

    treinamento(gerador,discriminador,gan,[smiles_arrays, classes_arrays],latent_dim,n_epocas=5000)