from tensorflow import keras
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from elmo_crf import CRF, ModelWithCRFLoss  

class ModelFactory(object):
    def __init__(self):
        initialized = True
        
    def resetSeeds(self):
        np.random.seed(1234567890)
        tf.random.set_seed(1234567890)
        
    #returns the sentiment extraction model with optimal parameters
    def getSentimentModel(self, max_seq_length, vocab_size, embedding_vectors, n_tags):

        inputs = Input(shape=(max_seq_length,))
        output = Embedding(vocab_size, 1024, weights=[embedding_vectors], input_length=max_seq_length, trainable=False)(inputs)

        output = Dropout(0.2)(output)
        output = Bidirectional(LSTM(units=128, return_sequences=True))(output)
        output = Dense(n_tags)(output)

        crf = CRF(dtype='float32')
        output = crf(output)
       
        base_model = keras.Model(inputs,output)
        model = ModelWithCRFLoss(base_model)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt)

        return model
    
    def getModifiersModel(self, max_seq_length, vocab_size, embedding_vectors, n_tags):
        
        inputs = Input(shape=(max_seq_length,))
        output = Embedding(vocab_size, 1024, weights=[embedding_vectors], input_length=max_seq_length, trainable=False)(inputs)

        output = Dropout(0.2)(output)
        output = Bidirectional(LSTM(units=128, return_sequences=True))(output)
        output = Dense(n_tags)(output)

        crf = CRF(dtype='float32')
        output = crf(output)
       
        base_model = keras.Model(inputs,output)
        model = ModelWithCRFLoss(base_model)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt)

        return model
    
    def getAspectsModel(self, max_seq_length, vocab_size, embedding_vectors, n_tags):

        inputs = Input(shape=(max_seq_length,))
        output = Embedding(vocab_size, 1024, weights=[embedding_vectors], input_length=max_seq_length, trainable=False)(inputs)

        output = Dropout(0.2)(output)
        output = Bidirectional(LSTM(units=128, return_sequences=True))(output)
        output = Bidirectional(LSTM(units=128, return_sequences=True))(output)
        output = Dense(n_tags)(output)

        crf = CRF(dtype='float32')
        output = crf(output)

        base_model = keras.Model(inputs,output)
        model = ModelWithCRFLoss(base_model)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt)
        
        return model

    def getModel(self,model_for, max_seq_length, vocab_size, embedding_vectors, n_tags):
        self.resetSeeds()
        print("generating model for", model_for)
        if model_for == "sentiments":
            return self.getSentimentModel(max_seq_length, vocab_size, embedding_vectors, n_tags)
        elif model_for == "modifiers":
            return self.getModifiersModel(max_seq_length, vocab_size, embedding_vectors, n_tags)
        else:
            return self.getAspectsModel(max_seq_length, vocab_size, embedding_vectors, n_tags)