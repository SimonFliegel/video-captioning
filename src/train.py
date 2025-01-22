import os
import json
import random

import joblib
import numpy as np
from flatbuffers.packer import float32
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import config
import video_processor
from model import create_models

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
        
        self.processor = video_processor.VideoProcessor(is_training=True)
        
        # processed data
        self.tokenizer = None
        
        # models
        self.combined_model = None
        self.encoder_model = None
        self.decoder_model = None
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
        random.shuffle(train_list) # randomize training data
        training_data = train_list[:int(len(train_list) * self.validation_split):]
        validation_data = train_list[:int(len(train_list) * self.validation_split)]
        
        train_feature_dir = os.path.join(self.train_path, 'features')
        for filename in os.listdir(train_feature_dir):
            features = self.processor.load_features(filename)
            if features.shape[0] < self.time_steps_encoder: # add padding if features of video with less than 80 frames
                padding = np.zeros((self.time_steps_encoder - features.shape[0], 4096))
                features = np.vstack((features, padding))
            self.x_data[filename[:-(len(".npy"))]] = features # remove .avi.npy to use video_id as key
            
        return training_data, validation_data

    def _load_dataset(self, training_data):
        def extract_video_data(training_data):
            """Extract video IDs and captions from training data."""
            video_ids, video_sequences = [], []
            for cap in training_data:
                video_sequences.append(cap[0])  # Captions
                video_ids.append(cap[1])       # Video IDs
            return video_ids, video_sequences

        def preprocess_sequences(sequences):
            """Tokenize and pad sequences."""
            tokenized = self.tokenizer.texts_to_sequences(sequences)
            return pad_sequences(
                tokenized, padding='post', truncating='post', maxlen=self.max_length
            )
    
        def generate_batches(video_ids, train_sequences):
            """Yield batches of encoder and decoder data."""
            encoder_input_data, decoder_input_data, decoder_target_data = [], [], []
            batch_count = 0
    
            for idx in range(len(train_sequences)):
                # Encoder input
                encoder_input_data.append(self.x_data[video_ids[idx]])
                # Decoder input and target
                y = to_categorical(train_sequences[idx], self.num_decoder_tokens)
                decoder_input_data.append(y[:-1])
                decoder_target_data.append(y[1:])
    
                batch_count += 1
                if batch_count == self.batch_size:
                    yield finalize_batch(encoder_input_data, decoder_input_data, decoder_target_data)
                    encoder_input_data, decoder_input_data, decoder_target_data = [], [], []
                    batch_count = 0
    
        def finalize_batch(encoder_inputs, decoder_inputs, decoder_targets):
            """Convert collected batch data to numpy arrays."""
            return (
                [
                    np.array(encoder_inputs, dtype=np.float32),
                    np.array(decoder_inputs, dtype=np.float32)
                ],
                np.array(decoder_targets, dtype=np.float32)
            )

        video_ids, video_sequences = extract_video_data(training_data)
        train_sequences = preprocess_sequences(video_sequences)
        for _ in range(2): # self.epochs
            yield from generate_batches(video_ids, train_sequences) # generator
    
    def _create_vocab_from_training_data(self, training_data):
        vocab = []
        for train in training_data:
            vocab.append(train[0])
        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        self.tokenizer.fit_on_texts(vocab)
        
    def _save_models(self):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.encoder_model.save(os.path.join(self.save_model_path, 'encoder_model.h5')) # encoder
        self.decoder_model.save_weights(os.path.join(self.save_model_path, 'decoder_model_weights.h5')) # decoder
        with open(os.path.join(self.save_model_path, 'tokenizer' + str(self.num_decoder_tokens)), 'wb') as file: # tokenizer
            joblib.dump(self.tokenizer, file)

    def _preprocess(self):
        training_data, validation_data = self._select_training_and_validation_data(os.path.join(self.train_path, 'captions.json'))
        self._create_vocab_from_training_data(training_data)
        return training_data, validation_data
        
    def setup(self):
        self._create_json_from_annotations()
        
    def train(self):
        self.combined_model, self.encoder_model, self.decoder_model = create_models()
        training_data, validation_data = self._preprocess()
        
        train = self._load_dataset(training_data)
        valid = self._load_dataset(validation_data)

        validation_steps = len(validation_data) // self.batch_size
        steps_per_epoch = len(training_data) // self.batch_size
        
        opt = Adam(learning_rate=self.learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto')
        
        self.combined_model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')
        self.combined_model.fit(train, validation_data=valid, validation_steps=validation_steps, epochs=1, steps_per_epoch=steps_per_epoch, callbacks=[early_stopping, reduce_lr])
        
        self._save_models()
    
    
if __name__ == '__main__':
    trainer = VideoCaptioningTrainer(config)
    # decoder.setup()
    trainer.train()
    

    