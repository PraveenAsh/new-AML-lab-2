from __future__ import print_function
import numpy as np
import csv, json
from zipfile import ZipFile
import os
import sys
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
from configuration import Config
config = Config()
GLOVE = None

#Greedy method to load just the first file in file list
print(os.path.join(config.GLOVE_DIR, config.GLOVE_ZIP_FILE[0]))
if not exists(os.path.join(config.GLOVE_DIR, config.GLOVE_ZIP_FILE[0])):
	zipfile = ZipFile(get_file(config.GLOVE_ZIP_FILE[0], config.GLOVE_DATA_URL))
	zipfile.extract(config.GLOVE_ZIP_FILE[0], path=KERAS_DATASETS_DIR)
	GLOVE = zipfile
	del zipfile

def load_dataset(filepath):
	question1 = []
	question2 = []
	labels = []
	with open(filepath, encoding = 'utf-8') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',')
		for row in reader:
			question1.append(row['question1'])
			question2.append(row['question2'])
			labels.append(row['is_duplicate'])
	return question1, question2, labels

def download_dataset():
	print('downloading dataset')

def maybe_download_dataset():
	noData = True
	question1 = []
	question2 = []
	labels = []
	for each in config.QOURA_DATASET_FILES:
		if exists(os.path.join(config.QOURA_DATASETS_DIR, each)):
			question1, question2, labels = load_dataset(os.path.join(config.QOURA_DATASETS_DIR, each))
			noData = False
	if(noData):
		download_dataset()
	return question1, question2, labels

def custom_tokenize(Q1, Q2, L):
	QUESTIONS = Q1 + Q2
	tokenizer = Tokenizer(num_words = config.MAX_NB_WORDS)
	tokenizer.fit_on_texts(QUESTIONS)
	Q1_sequences = tokenizer.texts_to_sequences(Q1)
	Q2_sequences = tokenizer.texts_to_sequences(Q2)
	word_index = tokenizer.word_index
	return Q1_sequences, Q2_sequences, word_index
	print(len(word_index))

def get_word_embeddings():
	embeddings_index = {}
	with open(os.path.join(config.GLOVE_DIR, config.GLOVE_FILE[0]), encoding='utf-8') as f:
		for line in f:
			values = line.split(' ')
			word = values[0]
			embedding = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = embedding
	print('Word embeddings: %d' % len(embeddings_index))
	return embeddings_index

def get_null_word_embedding(word_index, embeddings_index):
	nb_words = min(config.MAX_NB_WORDS, len(word_index))
	word_embedding_matrix = np.zeros((nb_words + 1, config.EMBEDDING_DIM))
	for word, i in word_index.items():
		if i > config.MAX_NB_WORDS:
			continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			word_embedding_matrix[i] = embedding_vector
	print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
	return nb_words, word_embedding_matrix

if __name__ == '__main__':
	QUESTION1, QUESTION2, LABELS = maybe_download_dataset()
	print(len(QUESTION1),len(QUESTION2),len(LABELS))
	Q1_SEQ, Q2_SEQ, word_index = custom_tokenize(QUESTION1, QUESTION2, LABELS)
	#All important variable
	Q1_SEQ = pad_sequences(Q1_SEQ, maxlen=config.MAX_SEQUENCE_LENGTH)
	Q2_SEQ = pad_sequences(Q2_SEQ, maxlen=config.MAX_SEQUENCE_LENGTH)
	labels = np.array(LABELS, dtype=int)
	embeddings_index = get_word_embeddings()
	nb_words, word_embedding_matrix = get_null_word_embedding(word_index, embeddings_index)


	X = np.stack((Q1_SEQ, Q2_SEQ), axis=1)
	y = labels
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SPLIT, random_state=config.RNG_SEED)
	Q1_train = X_train[:,0]
	Q2_train = X_train[:,1]
	Q1_test = X_test[:,0]
	Q2_test = X_test[:,1]

	#Model definition
	question1 = Input(shape=(config.MAX_SEQUENCE_LENGTH,))
	question2 = Input(shape=(config.MAX_SEQUENCE_LENGTH,))

	q1 = Embedding(nb_words + 1, 
		config.EMBEDDING_DIM, 
		weights=[word_embedding_matrix], 
		input_length=config.MAX_SEQUENCE_LENGTH, 
		trainable=False)(question1)
	q1 = TimeDistributed(Dense(config.EMBEDDING_DIM, activation='relu'))(q1)
	q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(config.EMBEDDING_DIM, ))(q1)

	q2 = Embedding(nb_words + 1, 
		config.EMBEDDING_DIM, 
		weights=[word_embedding_matrix], 
		input_length=config.MAX_SEQUENCE_LENGTH, 
		trainable=False)(question2)
	q2 = TimeDistributed(Dense(config.EMBEDDING_DIM, activation='relu'))(q2)
	q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(config.EMBEDDING_DIM, ))(q2)

	merged = concatenate([q1,q2])
	merged = Dense(200, activation='relu')(merged)
	merged = Dropout(config.DROPOUT)(merged)
	merged = BatchNormalization()(merged)
	merged = Dense(200, activation='relu')(merged)
	merged = Dropout(config.DROPOUT)(merged)
	merged = BatchNormalization()(merged)
	merged = Dense(200, activation='relu')(merged)
	merged = Dropout(config.DROPOUT)(merged)
	merged = BatchNormalization()(merged)
	merged = Dense(200, activation='relu')(merged)
	merged = Dropout(config.DROPOUT)(merged)
	merged = BatchNormalization()(merged)

	is_duplicate = Dense(1, activation='sigmoid')(merged)

	model = Model(inputs=[question1,question2], outputs=is_duplicate)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	reply = str(input('Do you want to train? '+' (y/n): ')).lower().strip()
	if not reply[0] == 'y':
		print('end')
	else:
		callbacks = [ModelCheckpoint(config.MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
		history = model.fit([Q1_train, Q2_train],
			y_train,
			epochs=config.NB_EPOCHS,
			validation_split=config.VALIDATION_SPLIT,
			verbose=2,
			batch_size=config.BATCH_SIZE,
			callbacks=callbacks)
		print('ended')
	reply = str(input('Do you want to test? '+' (y/n): ')).lower().strip()
	if not reply[0] == 'y':
		print('end')
	else:
		model.load_weights(config.MODEL_WEIGHTS_FILE)
		loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
		print('loss = {0:.4f}, accuracy = {1:.4f}'.format(loss, accuracy))