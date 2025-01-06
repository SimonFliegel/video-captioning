import functools
import operator
import os
import cv2
import time

import numpy as np
import video_processor

import config
import model


class VideoCaptionGenerator:
    def __init__(self, config):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability
        
        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = model.inference_model()
        self.inf_decoder_model = None
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        self.search_type = config.search_type
        self.num = 0
        
    def greedy_search(self, loaded_array):
        pass
    
    def beam_search(self, loaded_array):
        pass
    
    def test(self):
        test_files = os.listdir(os.path.join(self.test_path, 'videos'))
        for file in test_files:
            start = time.time()
            video_path = os.path.join(self.test_path, 'videos', file)
            features = video_processor.extract_features(video_path)
            caption = self.greedy_search(features)
            print(f'{file} - {caption}')
            end = time.time()
            print(f'Time taken: {end - start}')
            self.num += 1
            if self.num == 10:
                break
        
    def generate_caption(self, video_path):
        frames = self._extract_frames(video_path)
        frames = self._preprocess_frames(frames)
        caption = self._generate_caption(frames)
        return caption

    def _extract_frames(self, video_path):
        # Extract frames from video
        return frames

    def _preprocess_frames(self, frames):
        # Preprocess frames
        return frames

    def _generate_caption(self, frames):
        # Generate caption
        return caption