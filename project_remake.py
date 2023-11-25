# example of training an conditional gan on the fashion mnist dataset
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, \
                        Dropout, Embedding, Concatenate
from seq_encode.smiles import smiles_coder
import pandas as pd

# define the standalone discriminator model
def define_discriminator(in_shape, n_classes):
	print("Defining Discriminator...")
	print("In shape:", in_shape)
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	print(in_image.shape)
	print(li.shape)
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Dense(256)(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Dense(128)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	model.summary()
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes):
	print("Defining Generator...")
	# label input
	in_label = Input(shape=(1,))
	print(f"in_label: {in_label.shape}")
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	print(f"li: {li.shape}")
	# linear multiplication
	n_nodes = smiles_in_shape[0] * smiles_in_shape[1]/4
	li = Dense(n_nodes)(li)
	print(f"li: {li.shape}")
	# reshape to additional channel
	li = Reshape((int(smiles_in_shape[0]/2), int(smiles_in_shape[1]/2), 1))(li)
	print(f"li: {li.shape}")
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	print(f"in_lat: {in_lat.shape}")
	# foundation for 7x7 image
	n_nodes = int(smiles_in_shape[0] * smiles_in_shape[1]/4)
	gen = Dense(n_nodes)(in_lat)
	print(f"gen: {gen.shape}")
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((int(smiles_in_shape[0]/2), int(smiles_in_shape[1]/2), 1))(gen)
	print(f"gen: {gen.shape}")
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	print(f"merge: {merge.shape}")
	# upsample
	gen = Dense(128)(merge)
	print(f"gen: {gen.shape}")
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample
	gen = Dense(256)(gen)
	print(f"gen: {gen.shape}")
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Dense(smiles_in_shape[0] * smiles_in_shape[1], activation='tanh')(gen)
	print(f"out_layer: {out_layer.shape}")
	out_layer = Reshape(smiles_shape)(out_layer)
	print(f"out_layer: {out_layer.shape}")
	# define model
	model = Model([in_lat, in_label], out_layer)

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)

	model.summary()

	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	print("Defining GAN...")
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	print(gen_output.shape)
	print(gen_label.shape)
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load fashion mnist images
def load_real_samples():
	# load dataset
	trainX, trainy = smiles_arrays, classes_arrays
	
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X, trainy]

# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=5000, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, classes_in_shape[0])
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# save the generator model
	g_model.save('cgan_generator.h5')

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

	classes_hot_arrays = classes_df.to_numpy()

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

	smiles_shape = smiles_arrays.shape
	classes_shape = classes_arrays.shape

	smiles_in_shape = (smiles_shape[1], smiles_shape[2], 1)
	classes_in_shape = (classes_shape[1], 1)

	print("Smiles shape:", smiles_in_shape)
	print("Classes shape:", classes_in_shape)
	# size of the latent space
	latent_dim = 100
	# create the discriminator
	d_model = define_discriminator(in_shape=smiles_in_shape, n_classes=classes_in_shape[0])
	# create the generator
	g_model = define_generator(latent_dim, n_classes=classes_in_shape[0])
	# create the gan
	gan_model = define_gan(g_model, d_model)
	# load image data
	dataset = load_real_samples()
	# train model
	train(g_model, d_model, gan_model, dataset, latent_dim)