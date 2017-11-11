import io
import re
import pandas
import gensim
import numpy as np
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

def readCSV(in_file):
	dataframe = pandas.read_csv(in_file)
	return dataframe.values[:,3:6]

def normalize_sentence(sentence):
	#sentence = string.decode('utf-8')
	sentence = (str(sentence)).lower()

	# unit
	#weight
	sentence = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', sentence)	# e.g. 4kgs => 4 kg
	sentence = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', sentence)	 # e.g. 4kg => 4 kg
	sentence = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', sentence)	  # e.g. 4k => 4000
	#lenght
	sentence = re.sub(r"(\d+)km ", lambda m: m.group(1) + ' km ', sentence)
	sentence = re.sub(r"(\d+)cm ", lambda m: m.group(1) + ' cm ', sentence)
	sentence = re.sub(r"(\d+)m ", lambda m: m.group(1) + ' m ', sentence)
	#data
	sentence = re.sub(r"(\d+)gb ", lambda m: m.group(1) + ' gb ', sentence)
	#money
	sentence = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', sentence)
	sentence = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', sentence)

	# acronym
	sentence = re.sub(r"can\'t", "can not", sentence)
	sentence = re.sub(r"cannot", "can not ", sentence)
	sentence = re.sub(r"what\'s", "what is", sentence)
	sentence = re.sub(r"\'ve ", " have ", sentence)
	sentence = re.sub(r"n\'t", " not ", sentence)
	sentence = re.sub(r"i\'m", "i am ", sentence)
	sentence = re.sub(r"\'re", " are ", sentence)
	sentence = re.sub(r"\'d", " would ", sentence)
	sentence = re.sub(r"\'ll", " will ", sentence)
	sentence = re.sub(r" e mail ", " email ", sentence)
	sentence = re.sub(r" e \- mail ", " email ", sentence)
	sentence = re.sub(r"e-mail", " email ", sentence)
	sentence = re.sub(r",000", '000', sentence)
	sentence = re.sub(r"\'s", " ", sentence)
	
	# punctuation
	sentence = re.sub(r"\+", " + ", sentence)
	sentence = re.sub(r"'", " ", sentence)
	sentence = re.sub(r"\"", "", sentence)
	sentence = re.sub(r"-", " ", sentence)
	sentence = re.sub(r"/", " / ", sentence)
	sentence = re.sub(r"\\", " \ ", sentence)
	sentence = re.sub(r"=", " = ", sentence)
	sentence = re.sub(r"\^", " ^ ", sentence)
	sentence = re.sub(r":", " : ", sentence)
	sentence = re.sub(r"\.", "", sentence)
	sentence = re.sub(r",", " ", sentence)
	sentence = re.sub(r"\?", "", sentence)
	sentence = re.sub(r"!", "", sentence)
	sentence = re.sub(r"\"", " \" ", sentence)
	sentence = re.sub(r"&", " & ", sentence)
	sentence = re.sub(r"\|", " | ", sentence)
	sentence = re.sub(r";", " ; ", sentence)
	sentence = re.sub(r"\(", " ( ", sentence)
	sentence = re.sub(r"\)", " ) ", sentence)

	# symbol replacement
	sentence = re.sub(r"&", " and ", sentence)
	sentence = re.sub(r"\|", " or ", sentence)
	sentence = re.sub(r"\\", " or ", sentence)
	sentence = re.sub(r"/", " or ", sentence)
	sentence = re.sub(r"=", " equal ", sentence)
	sentence = re.sub(r"\+", " plus ", sentence)
	sentence = re.sub(r"₹", " rs ", sentence)
	sentence = re.sub(r"\$", " dollar ", sentence)
	sentence = re.sub(r"%", " percent ", sentence)
	while '  ' in sentence:
		sentence = re.sub(r"  ", " ", sentence)
	return sentence

def normalize_sentence_list(list_sentence):
	for i in range (0,len(list_sentence)):
		list_sentence[i] = normalize_sentence(list_sentence[i])
	return list_sentence

def normalize_sentence_dataset(dataset):
	dataset[:,0] = normalize_sentence_list(dataset[:,0])
	dataset[:,1] = normalize_sentence_list(dataset[:,1])
	return dataset

def lemmatize_sentence_dataset(dataset):
	lemmatizer = WordNetLemmatizer()
	for idx_col in range(0,2):
		for idx_row in range (0,len(dataset[:,0])):  
			token_list = dataset[idx_row,idx_col].split(" ")
			dataset[idx_row,idx_col] = ""
			for i in range(0,len(token_list)):
				dataset[idx_row,idx_col] += lemmatizer.lemmatize(token_list[i])+" "
	return dataset

def remove_stop_word(sentence):
	for stopword in stopwords.words('english'):
			sentence = re.sub(r" "+stopword+" ", " ", sentence)
	return sentence

def remove_shared_word(list_sentence):
	duplicate = []
	not_duplicate = []

	for sentence in list_sentence:
		list_of_same_word = []
		try:
			sentence_0 = word_tokenize(sentence[0])
			sentence_1 = word_tokenize(sentence[1])
			for word in sentence_0:
				if word in sentence_1:
					list_of_same_word.append(word)
			if sentence[2] == 0:
				not_duplicate.extend(list_of_same_word)
			else:
				duplicate.extend(list_of_same_word)
		except:
			pass
	shared_word = []
	same_word_duplicate = Counter(duplicate).most_common(500)
	same_word_not_duplicate = Counter(not_duplicate).most_common(500)
	for key, count in same_word_duplicate:
		for index, value in enumerate(same_word_not_duplicate):
			if key == value[0]:
				shared_word.append(key)
				del same_word_not_duplicate[index]
				break;
	
	
	for sentence in list_sentence:
		for word in shared_word:
			sentence[0] = re.sub(r" "+word+" ", " ", sentence[0])
			sentence[1] = re.sub(r" "+word+" ", " ", sentence[1])

	return list_sentence
		
def get_avg_vector(sentence, model, num_features=300):
	sentence = sentence.strip().split(' ')
	sentence_avg_vec = np.zeros((num_features,),dtype="float32")
	n_vocab_word = 0
	for word in sentence:
		if (word in model.wv.vocab):
			n_vocab_word += 1
			sentence_avg_vec = np.add(sentence_avg_vec, model[word])
		elif (word.title() in model.wv.vocab):
			n_vocab_word += 1
			sentence_avg_vec = np.add(sentence_avg_vec, model[word.title()])
		elif (word.upper() in model.wv.vocab):
			n_vocab_word += 1
			sentence_avg_vec = np.add(sentence_avg_vec, model[word.upper()])
	if (n_vocab_word != 0):
		sentence_avg_vec = np.divide(sentence_avg_vec, n_vocab_word)
	return sentence_avg_vec

def get_number_count(sentence):
	count = 0
	for word in sentence.strip().split(' '):
		if (word.isdigit()):
			count += 1
	return count
	
def get_oov_count(sentence, model):
	count = 0
	sentence = sentence.strip().split(' ')
	for word in sentence:
		if ((word not in model.wv.vocab) and (word.title() in model.wv.vocab) and (word.upper() in model.wv.vocab)):
			count += 1
	return count
	
def get_features(data, model):
	q1 = get_avg_vector(data[0], model)
	q2 = get_avg_vector(data[1], model)
	n_w_q1 = len(data[0].strip().split(' '))
	n_w_q2 = len(data[1].strip().split(' '))
	return {
		'q1': q1,
		'q2': q2,
		'cosine_similarity': np.dot(q1, q2)/(np.linalg.norm(q1)* np.linalg.norm(q2)),
		'n_words_q1': n_w_q1,
		'n_words_q2': n_w_q2,
		'diff_words': abs(n_w_q1-n_w_q2),
		'first_word_q1': data[0].split(' ')[0],
		'first_word_q2': data[1].split(' ')[0],
		'number_count_q1': get_number_count(data[0]),
		'number_count_q2': get_number_count(data[1]),
		'oov_count_q1': get_oov_count(data[0], model),
		'oov_count_q2': get_oov_count(data[1], model),
	}

dataset = readCSV('train.csv')
f_dataset = remove_shared_word(dataset[0:100])
f_dataset = normalize_sentence_dataset(f_dataset)
f_dataset = lemmatize_sentence_dataset(f_dataset)

filename = 'GoogleNews-vectors-negative300-SLIM.bin' #https://github.com/eyaler/word2vec-slim
model = KeyedVectors.load_word2vec_format(filename, binary=True)
print(get_features(f_dataset[0], model))
