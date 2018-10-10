import os
class Config():
	def __init__(self):
		self.QOURA_DATASETS_DIR = 'qoura'
		self.GLOVE_DIR = 'glove'
		self.GLOVE_DATA_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
		self.QOURA_DATASET_FILES = ['data-100-qoura.csv','data-1000-qoura.csv','data-10000-qoura.csv','data-20000-qoura.csv']
		self.GLOVE_ZIP_FILE = ['glove.6B.zip',' glove.42B.300d.zip','glove.840B.300d.zip',' glove.twitter.27B.zip','GoogleNews-vectors-negative300.bin.gz']
		self.GLOVE_FILE = ['glove.6B.100d.txt',' glove.42B.300d.txt','glove.840B.300d.txt',' glove.twitter.27B.txt','GoogleNews-vectors-negative300.txt']
		self.MAX_NB_WORDS = 200000
		self.MAX_SEQUENCE_LENGTH = 25
		self.EMBEDDING_DIM = 100
		self.VALIDATION_SPLIT = 0.1
		self.TEST_SPLIT = 0.1
		self.RNG_SEED = 13371447
		self.NB_EPOCHS = 500
		self.DROPOUT = 0.1
		self.BATCH_SIZE = 32
		self.MODEL_WEIGHTS_FILE = os.path.join('models', 'modelsquestion_pairs_weights.h5')
		self.ENABLE_GPU = False

if __name__ == '__main__':
	config = Config()
	print(config.QOURA_DATASETS_DIR)