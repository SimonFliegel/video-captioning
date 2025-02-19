import functools
import operator
import os
import shutil
import random
import cv2
import time

import numpy as np
import video_processor

import config
import model

class VideoCaptionInference:
    def __init__(self, conf):
        self.test_size = 10
        self.test_path = conf.test_path
        self.train_path = conf.train_path
        
        self.latent_dim = conf.latent_dim
        self.num_encoder_tokens = conf.num_encoder_tokens
        self.num_decoder_tokens = conf.num_decoder_tokens
        self.time_steps_encoder = conf.time_steps_encoder
        self.max_length = conf.max_length
        self.max_probability = conf.max_probability
        self.decode_sequence = []
        
        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = model.load_inference_models()
        self.save_model_path = conf.save_model_path
        self.test_path = conf.test_path
        self.search_type = conf.search_type
        
        self.video_processor = video_processor.VideoProcessor(is_training=False)
        
    def _select_testing_data(self):
        train_videos_path = os.path.join(self.train_path, 'videos')
        test_videos_path = os.path.join(self.test_path, 'videos')
        test_features_path = os.path.join(self.test_path, 'features')
        
        all_videos = os.listdir(train_videos_path)
        test_videos = random.sample(all_videos, self.test_size)
        
        if os.path.isdir(test_videos_path):
            shutil.rmtree(test_videos_path)
        if os.path.isdir(test_features_path):
            shutil.rmtree(test_features_path)
        os.makedirs(test_videos_path)
            
        with open(os.path.join(self.test_path, 'testing_id.txt'), 'w') as testing_file:
            for video in test_videos:
                video_id = video.split('.')[0]
                testing_file.write(video_id + '\n')
                shutil.copy(os.path.join(train_videos_path, video), test_videos_path)
                
        self.video_processor.create_features()
        
    def _greedy_search(self, loaded_array):
        """
        Always picks the token with the highest probability as the next word.
        :param loaded_array: after extracting features from videos: 
        :return: the caption:
        """
        inv_map = self._index_to_word()
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(-1, self.time_steps_encoder, self.num_encoder_tokens))
        target_sequence = np.zeros((1, 1, self.num_decoder_tokens))
        sentence = ''
        target_sequence[0, 0, self.tokenizer.word_index['bos']] = 1
        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_sequence] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)
            if y_hat == 0:
                continue
            if inv_map[y_hat] is None:
                break
            else: 
                sentence += inv_map[y_hat] + ' '
                target_sequence = np.zeros((1, 1, self.num_decoder_tokens))
                target_sequence[0, 0, y_hat] = 1
        return ' '.join(sentence.split()[:-1])
    
    def _decode_sequence_to_beam_search(self, input_sequence):
        states_value = self.inf_encoder_model.predict(input_sequence)
        target_sequence = np.zeros((1, 1, self.num_decoder_tokens))
        target_sequence[0, 0, self.tokenizer.word_index['bos']] = 1
        self._beam_search(target_sequence, states_value, [], [], 0)
        return self.decode_sequence
    
    def _beam_search(self, target_sequence, state_value, prob, path, lens):
        max_len = self.max_length + 2 # <bos> and <eos>
        beam_width = 2 # nodes to keep
        output_tokens, h, c = self.inf_decoder_model.predict([target_sequence] + state_value)
        output_tokens = output_tokens.reshape(self.num_decoder_tokens)
        sampled_token_index = output_tokens.argsort()[-beam_width:][::-1]
        states_value = [h, c]
        
        for i in range(beam_width):
            if sampled_token_index[i] == 0:
                sampled_char = ''
            else:
                sampled_char = {values: keys for keys, values in self.tokenizer.word_index.items()}.get(sampled_token_index[i], '')

            if sampled_char != 'eos' and lens <= max_len:
                p = output_tokens[sampled_token_index[i]]
                if sampled_char == '':
                    p = 1
                prob_new = list(prob)
                prob_new.append(p)
                path_new = list(path)
                path_new.append(sampled_char)
                target_sequence = np.zeros((1, 1, self.num_decoder_tokens))
                target_sequence[0, 0, sampled_token_index[i]] = 1
                self._beam_search(target_sequence, states_value, prob_new, path_new, lens + 1)
            else:
                p = output_tokens[sampled_token_index[i]]
                prob_new = list(prob)
                prob_new.append(p)
                p = functools.reduce(operator.mul, prob_new, 1) # multiply all probabilities
                if p > self.max_probability:
                    self.decode_sequence = path
                    self.max_probability = p

    def _index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word
                    
    def _decoded_sentence_tuning(self, decoded_sentence):
        decode_str = []
        filter_string = ['bos', 'eos']
        uni_gram = {}
        last_string = ''
        for i, c in enumerate(decoded_sentence):
            if c in uni_gram:
                uni_gram[c] += 1
            else:
                uni_gram[c] = 1
            if last_string == c and i > 0:
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if i > 0:
                last_string = c
        return decode_str
    
    def _get_test_data(self):
        x_test = []
        x_test_filename = []
        
        with open(os.path.join(self.test_path, 'testing_id.txt')) as testing_file:
            lines = testing_file.readlines()
            for filename in lines:
                filename = filename.strip()
                f = np.load(os.path.join(self.test_path, 'features', filename + '.npy'))
                x_test.append(f)
                x_test_filename.append(filename[:-len('.npy')])
            return x_test, x_test_filename
        
    def test(self):
        self._select_testing_data()
        x_test, x_test_filename = self._get_test_data()
        
        with open(os.path.join(self.test_path, 'test_%s.txt' % self.search_type), 'w') as file:
            for i, x in enumerate(x_test):
                file.write(x_test_filename[i] + ',')
                start = time.time()
                if self.search_type == 'greedy':
                    decoded_sentence = self._greedy_search(x.reshape(-1, self.time_steps_encoder, self.num_encoder_tokens))
                    file.write(decoded_sentence + ',{:.2f}'.format(time.time() - start))
                else:
                    decoded_sentence = self._decode_sequence_to_beam_search(x.reshape(-1, self.time_steps_encoder, self.num_encoder_tokens))
                    decode_str = self._decoded_sentence_tuning(decoded_sentence)
                    for d in decode_str:
                        file.write(d + ' ')
                    file.write(',{:.2f}'.format(time.time() - start))
                file.write('\n')
                
                self.decode_sequence = []
                self.max_probability = -1

    def predict_realtime(self, video_path):
        features = self.video_processor.extract_features(video_path)
        if self.search_type == 'greedy':
            caption = self._greedy_search(features.reshape(-1, self.time_steps_encoder, self.num_encoder_tokens))
        else:
            decoded_sentence = self._decode_sequence_to_beam_search(features.reshape(-1, self.time_steps_encoder, self.num_encoder_tokens))
            decoded_sentence = self._decoded_sentence_tuning(decoded_sentence)
            caption = ' '.join(decoded_sentence)
        # caption = '[' + ' '.join(caption.split()[:-1]) + ']' # remove <eos> token
        print(caption)

        original = cv2.VideoCapture(video_path)
        captioned = cv2.VideoCapture(video_path)
        while original.isOpened():
            ret_original, frame_original = original.read()
            ret_captioned, frame_captioned = captioned.read()
            if ret_original:
                image_show = cv2.resize(frame_original, (480, 300))
                cv2.imshow('Original', image_show)
            if ret_captioned:
                image_show = cv2.resize(frame_captioned, (480, 300))
                cv2.putText(image_show, caption, (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_4)
                cv2.imshow('Captioned', image_show)
            else:
                break

            key = cv2.waitKey(1)
            if key == 27: # ESC
                break

        original.release()
        captioned.release()
        cv2.destroyAllWindows()



    


if __name__ == '__main__':
    vc = VideoCaptionInference(config)
    vc.test()