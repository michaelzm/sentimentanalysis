import json
import numpy as np
import pandas as pd
import os

from tensorflow.keras import Model
import tensorflow as tf

from tensorflow import keras
import zipfile
import pickle
from sklearn.utils import class_weight

from elmo_dataset_joiner import DatasetWorker, VocabularyWorker
from elmo_performance import PerformanceViewer
from model_factory import ModelFactory

np.random.seed(1234567890)
tf.random.set_seed(1234)       

def resetSeeds():
    np.random.seed(1234567890)
    tf.random.set_seed(1234567890)    
    
    
#tensorflow stuff
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tensorflow stuff in case mem overflows or something like that
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

## first step is to create the different datasets for sentiments, aspects and modifiers
extracts = ["sentiments", "aspects", "modifiers"]

vw= VocabularyWorker()

#data_laptop contains all reviews inclusive the test data
filename = r'data_laptop_absa.json'

#Embedding list from pickle file
embedding = pickle.load( open("elmo_embedding.pkl", "rb" ) )

##test_id_list contains all the test data which we keep seperate from the other data
test_id_list = pickle.load(open("test_id_list.pkl", "rb"))

with open(filename,'r', encoding='utf8') as infile:
    review_data = json.load(infile)

#confusion matrix will get stored inside here
# idx 0 -> sentiments, idx 1 -> aspects, idx 2 -> modifiers
results = []
filename = r'data_laptop_absa.json'

#gridsearch evaluated 5 epochs for sentiments and modifiers and 4 for aspects
epochs_parameter = {"sentiments":5, "modifiers":5, "aspects":4}

for extract in extracts:
    extraction_of = extract
    print("start building model for ", extraction_of)
    
    max_seq_length = 100
    ds = DatasetWorker(review_data)
    
    #tokenize data
    ds.applyPreprocessing()
    #todo set extraction_of correct
    ds.setExtractionOf(extraction_of)
    
    #options for splitDataset = every_review, every_review_without_uncertain
    #what we get is for example 
    #tokens[0]  [['computer', 'works', 'great', '.'], 1, 'train']
    #labels[0] ['O', 'O', 'O', 'O']
    tokens, labels =ds.splitDataset("every_review_without_uncertain", test_id_list)
    

    #build vocab and add embedding#
    #with elmo, every token of each sentence gets its individual elmo vecotr

    # we have a total of 54352 vectors
    vocab_size = len (embedding)

    #each vector has 1024 dimensions (smallest elmo dataset)
    embed_size = 1024

    embedding_vectors = np.zeros((vocab_size, embed_size))

    #embedding-vectors for Embedding layer 
    for i,embedding_element in enumerate(embedding):
        #we only extract the vector data
        token_vector = embedding_element[1]
        embedding_vectors[i] = token_vector
        
    #extract labelclasses
    all_labelclasses = set()
    for row in labels:
        all_labelclasses.update(row)
    all_labelclasses=list(all_labelclasses)
    all_labelclasses.sort()

    labelclass_to_id = dict(zip(all_labelclasses,list(range(len(all_labelclasses)))))
  
    
    #define number of labelclasses as n_tags
    n_tags = len(list(labelclass_to_id.keys()))

    #max sequence length defines the length each review should get padded to
    max_seq_length = 100

    #convert token and label to ids according to the embedding and split into train and test
    #test_tokens is needed for calculation of the confusion matrix
    #x_train,x_test, y_train, y_test are for training and testing respectively
    # train_tokens and test_tokens are used for better analyzing the data because they dont contain padded elements
    #because y data changes per extraction_of we have to let this run through every time
    x_train, x_test, y_train, y_test , train_tokens, test_tokens = vw.convert_tokens_labels_list_to_ids_list(tokens, labels, embedding, max_seq_length, labelclass_to_id, n_tags)
    
    ##get the specific model
    resetSeeds()
    model_factory = ModelFactory()
    model = model_factory.getModel(extraction_of, max_seq_length, vocab_size, embedding_vectors, n_tags)
    
    history = model.fit(
    x_train, y_train,
    batch_size=16,
    validation_split = 0.2,
    verbose = 1,
    epochs=epochs_parameter[extraction_of])
    
    
      #to evaluate metrics
    performance = PerformanceViewer(labelclass_to_id)
    
    results.append(performance.classicEvalCrf(model,x_test,y_test, test_tokens))
    
# display the results
overall_f1 = 0
for result in results:
    display(result[0])
    print("precision: ","{:.3f}".format(result[1]))
    print("recall: ","{:.3f}".format(result[2]))
    print("f1: ","{:.3f}".format(result[3]))
    overall_f1 += result[3]
    
print("overall f1-score ", "{:.3f}".format((overall_f1)/3))