import io
import re
import pandas
from nltk.stem import WordNetLemmatizer
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

dataset = readCSV('train.csv')
dataset = normalize_sentence_dataset(dataset[:10,:2])
print(lemmatize_sentence_dataset(dataset))
#print(normalize_sentence('ahok beratnya 10kg&suka tulis\email kaya gini "e-mail"'))

