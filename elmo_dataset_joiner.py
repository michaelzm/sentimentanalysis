#all work to prepare the dataset is done inside DatasetWorker
import numpy as np
from tqdm import tqdm 
import os
import zipfile
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

np.random.seed(1234567890)

class DatasetWorker(object):
    #init the worker with the dataset (in our case its the example_data)
    def __init__(self, _dataset):
        self.dataset = _dataset
        self.split_size = 0
        
        #default extraction is sentiments
        self.extraction_of = "sentiments"
        
        self.train_tokens = []
        self.test_tokens = []
        
        self.train_labels = list()
        self.test_labels = list()
        self.train_labels_uncertainty = list()
        
        self.labels = []
        self.tokens = []
        
    
    #tokenize the dataset
    def tokenizeDataset(self):
        for i,(k,v) in tqdm(enumerate(self.dataset.items()), desc="tokenize dataset"):
            tokens = v.get("tokens")
            tokens = [token.lower() for token in tokens]
            self.dataset[k]["tokens"] = tokens
                
    # define here extraction of sentiment, aspect or modifier
    # only once at a time
    def setExtractionOf(self, to_extract):
        self.extraction_of = to_extract
        
    ##apply preprocessingg (only tokenizing)
    def applyPreprocessing(self):
        self.tokenizeDataset()
    
    
    # param split_by 
    # 
    # 1. every_review -> treat every review as its own and dont merge labels
    # 2. every_review_without_uncertain -> only use the users snetence if he didnt label any difficulty / uncertainty
    def splitDataset(self, split_by, test_ids):
        sentence_id= 1
        uncertainty_str=self.extraction_of+"_uncertainty"
        #difficulty_str=self.extraction_of+"_difficulty"
        
        for d_idx, (k,v) in tqdm(enumerate(self.dataset.items()), desc="split dataset labels"):
            curr_users = [s for s in v.keys() if s != "tokens"]         
            
            #use every review on its own as input
            if split_by == "every_review_without_uncertain":
                uncertain = False
                for usr in curr_users:
                    # iterate labeled sentences and look if there were any difficulties or uncertainties marked
                    for s_u in v[usr][uncertainty_str]:
                        if s_u != "O":
                            uncertain = True
                    if uncertain == False:
                        self.labels.append(v[usr][self.extraction_of])
                        if k in test_ids:
                            token_id_list = [v['tokens'], sentence_id, "test"]
                        else:
                            token_id_list = [v['tokens'], sentence_id, "train"]
                        self.tokens.append(token_id_list)
                        
                sentence_id += 1
                    
            elif split_by == "every_review":
                #use every users review as own label and token data entry
                
                for usr in curr_users:
                    self.labels.append(v[usr][self.extraction_of])
                    if k in test_ids:
                        token_id_list = [v['tokens'], sentence_id, "test"]
                    else:
                        token_id_list = [v['tokens'], sentence_id, "train"]
                    self.tokens.append(token_id_list)
                    
                sentence_id +=1
            else:
                print(split_by)
                raise ValueError('split_by operator not defined!')
                
        token_list = self.tokens
        label_list = self.labels
        
        return token_list, label_list
    
    # returns all used tokens of current extraction
    def getUsedLabels(self):
        used_lab = set()
        for t in self.train_labels:
            used_lab.update(t)
            
        return used_lab
    
                    

#does vocabulary, token ids and embedding for vocabulary
class VocabularyWorker(object):
    
    #convert token and labels to ids according to the embedding
    def convert_tokens_labels_list_to_ids_list(self, tokens_list, labels_list,embedding, max_seq_length,labelclass_to_id, n_tags):
        
        #lists which get returned 
        train_token_ids_list, train_label_ids_list = [], []
        test_token_ids_list, test_label_ids_list = [], []
        train_token_ids_list_no_0, test_token_ids_list_no_0 = [], []
        
        #index for getting the right token id from the embedding
        
        for index in tqdm(range(len(tokens_list)), desc="Converting tokens & labels to ids "):
            
            #lists contain token and label from one sentence
            tokens = tokens_list[index]
            labels = labels_list[index]   
            token_ids = []
            
            is_fail = False
            
            i=1
            while i<54353:
                if tokens[1]==embedding[i][2]:
                    break
                i+=1
                
            token_counter = 0
            
            for token in tokens[0]:
                j = i + token_counter
                if token == embedding[j][0]:
                    token_ids.append(j)
                else:
                    print("fail", j)
                    is_fail = True
                    break
                
                
                token_counter += 1
                
            if is_fail==True:
                break
            label_ids = [labelclass_to_id[label] for label in labels]
            
            token_ids_no_0 = token_ids[0:max_seq_length]
            token_ids_short = token_ids[0:max_seq_length]
            label_ids_short = label_ids[0:max_seq_length]
                       
            # Zero-pad up to the sequence length.
            while len(token_ids_short) < max_seq_length:
                token_ids_short.append(0)
                label_ids_short.append(0)
                
            if tokens[2]=="test":
                test_token_ids_list.append(token_ids_short)
                test_label_ids_list.append(label_ids_short)
                test_token_ids_list_no_0.append(token_ids_no_0)
            else:    
                train_token_ids_list.append(token_ids_short)
                train_label_ids_list.append(label_ids_short)
                train_token_ids_list_no_0.append(token_ids_no_0)
            
            
        #removed to_categorical
        return (
            np.array(train_token_ids_list),
            np.array(test_token_ids_list),
            np.array(train_label_ids_list),
            np.array(test_label_ids_list),
            train_token_ids_list_no_0,
            test_token_ids_list_no_0
        )
