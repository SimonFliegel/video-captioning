import os
import json
import random

import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import config
from src.config import num_decoder_tokens


class VideoCaptioningTrainer:
    
    def __init__(self, conf):
        self.train_path = conf.train_path
        self.test_path = conf.test_path
        self.min_length = conf.min_length
        self.max_length = conf.max_length
        self.batch_size = conf.batch_size
        self.validation_split = conf.validation_split
        self.learning_rate = conf.learning_rate
        self.epochs = conf.epochs
        self.latent_dim = conf.latent_dim
        self.num_encoder_tokens = conf.num_encoder_tokens
        self.num_decoder_tokens = conf.num_decoder_tokens
        self.time_steps_encoder = conf.time_steps_encoder
        self.time_steps_decoder = conf.time_steps_decoder
        self.x_data = {}
        
        # processed data
        self.tokenizer = None
        
        # models
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = conf.save_model_path
        

    def _create_json_from_annotations(self):
        annotations = {}
        with open(os.path.join(self.train_path, 'captions.txt'), 'r') as file:
            for line in file:
                video_id, caption = line.split(' ', 1)
                caption = caption.strip() # remove line break
                if video_id not in annotations:
                    annotations[video_id] = []
                annotations[video_id].append(caption)
        json_data = [{"id": video_id, "captions": captions} for video_id, captions in annotations.items()]
        with open(os.path.join(self.train_path, 'captions.json'), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    def _select_training_and_validation_data(self, json_file_path):
        with open(json_file_path, 'r') as data_file:
            y_data = json.load(data_file)
        train_list = []
        for y in y_data:
            for caption in y['captions']:
                caption = '<bos> ' + caption + ' <eos>'
                word_count = len(caption.split(' '))
                if self.min_length <= word_count <= self.max_length:
                    train_list.append([caption, y['id']])
        print(len(train_list))
        random.shuffle(train_list) # randomize training data
        training_data = train_list[:int(len(train_list) * self.validation_split):]
        validation_data = train_list[:int(len(train_list) * self.validation_split)]
        train_feature_dir = os.path.join(self.train_path, 'features')
        for filename in os.listdir(train_feature_dir):
            f = np.load(os.path.join(train_feature_dir, filename), allow_pickle=True)
            self.x_data[filename[:-4]] = f
        return training_data, validation_data
    
    def _load_dataset(self, training_data):
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []
        video_id = []
        video_seq = []
        for idx, cap in enumerate(training_data):
            caption = cap[0]
            video_id.append(cap[1])
            video_seq.append(caption)
        train_sequences = self.tokenizer.texts_to_sequences(video_seq)
        train_sequences = np.array(train_sequences)
        train_sequences = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=self.max_length)
        file_size = len(train_sequences)
        n = 0
        for i in range(self.epochs):
            for idx in range(0, file_size):
                n += 1
                encoder_input_data.append(self.x_data[video_id[idx]])
                y = to_categorical(train_sequences[idx], self.num_decoder_tokens)
                decoder_input_data.append(y[:-1])
                decoder_target_data.append(y[1:])
                if n == self.batch_size:
                    encoder_input = np.array(encoder_input_data)
                    decoder_input = np.array(decoder_input_data)
                    decoder_target = np.array(decoder_target_data)
                    encoder_input_data = []
                    decoder_input_data = []
                    decoder_target_data = []
                    n = 0
                    yield [encoder_input, decoder_input], decoder_target
    
    def _create_vocab_from_training_data(self, training_data):
        vocab = []
        for train in training_data:
            vocab.append(train[0])
        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        self.tokenizer.fit_on_texts(vocab)
        
    def _save_model(self):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.encoder_model.save(os.path.join(self.save_model_path, 'encoder_model.h5'))
        self.decoder_model.save_weights(os.path.join(self.save_model_path, 'decoder_model_weights.h5'))
        with open(os.path.join(self.save_model_path, 'tokenizer' + str(self.num_decoder_tokens)), 'wb') as file:
            joblib.dump(self.tokenizer, file)
        
    def setup(self):
        self._create_json_from_annotations()
        
    def preprocess(self):
        training_data, validation_data = self._select_training_and_validation_data(os.path.join(self.train_path, 'captions.json'))
        self._create_vocab_from_training_data(training_data)
        return training_data, validation_data
        
    
    def train(self):
        encoder_inputs = Input(shape=(config.time_steps_encoder, config.num_encoder_tokens), name='encoder_inputs')
        encoder_lstm = LSTM(config.latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
        _, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        decoder_inputs = Input(shape=(self.time_steps_decoder, num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(config.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()
        training_data, validation_data = self.preprocess()
        
        train = self._load_dataset(training_data)
        valid = self._load_dataset(validation_data)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')
        
        opt = Adam(learning_rate=self.learning_rate)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto')
        model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')
        
        validation_steps = len(validation_data) // self.batch_size
        steps_per_epoch = len(training_data) // self.batch_size
        
        model.fit(train, validation_data=valid, validation_steps=validation_steps, epochs=self.epochs, steps_per_epoch=steps_per_epoch, callbacks=[early_stopping, reduce_lr])
        
        self.encoder_model.summary()
        self.decoder_model.summary()
        
        self._save_model()
    
    
if __name__ == '__main__':
    decoder = VideoCaptioningTrainer(config)
    # decoder.setup()
    decoder.train()
    
    

    