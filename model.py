from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy
from keras import optimizers
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
import feature_extract

def classlabaelencoder(y):
	encoder = LabelEncoder()
	encoder.fit(y)
	y_encoded = encoder.transform(y)
	return np_utils.to_categorical(y_encoded) 
def main():

	
	dataset = feature_extract.readCSV('train.csv')
	used = dataset[5000:15000]
	f_dataset = feature_extract.remove_shared_word(used)
	f_dataset = feature_extract.normalize_sentence_dataset(f_dataset)
	f_dataset = feature_extract.lemmatize_sentence_dataset(f_dataset)

	filename = 'GoogleNews-vectors-negative300-SLIM.bin' #https://github.com/eyaler/word2vec-slim
	w2v_model = KeyedVectors.load_word2vec_format(filename, binary=True)

	x_train = []
	y_train = []
	for data in f_dataset:
		x_train.append(feature_extract.get_features_list(data,w2v_model))
		y_train.append(data[2])

	model = Sequential()
	model.add(Dense(input_dim=len(x_train[0]), units=150, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=2, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizers.Adagrad(lr=0.001), metrics=['accuracy'])
	model.fit(x_train, classlabaelencoder(y_train), epochs=50000, validation_split=0.2)


if __name__ == "__main__":
	main()
