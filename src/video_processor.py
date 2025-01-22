import os
import cv2
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
import config

class VideoProcessor:
    def __init__(self, is_training=True):
        self.frame_count = 80
        self.data_path = config.train_path if is_training else config.test_path
        self.model = self._load_model()

    def _load_model(self):
        model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        return Model(inputs=model.input, outputs=out)
    
    def _get_frames(self, path_to_video):
        cap = cv2.VideoCapture(path_to_video)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            frames.append(frame)
        cap.release()
        cv2.destroyAllWindows()
        return frames
    
    def extract_features(self, path_to_video):
        frames = self._get_frames(path_to_video)
        if len(frames) < self.frame_count:
            samples = np.arange(len(frames))
        else:
            samples = np.round(np.linspace(0, len(frames) - 1, self.frame_count)).astype(int) # select 80 frames
        images = np.zeros((len(samples), 224, 224, 3))
        for i in range(len(samples)):
            images[i] = cv2.resize(frames[i], (224, 224))
        images = np.array(images)
        fc_features = self.model.predict(images, batch_size=128)
        image_features = np.array(fc_features)
        return image_features
    
    def save_features(self, path_to_video, features):
        feat_dir = os.path.join(self.data_path, 'features')
        if not os.path.isdir(feat_dir):
            os.makedirs(feat_dir)

        base_name = os.path.splitext(os.path.basename(path_to_video))[0]
        outfile = os.path.join(feat_dir, base_name + '.npy')
        np.save(outfile, features)
        
    def load_features(self, path_to_video):
        feat_dir = os.path.join(self.data_path, 'features')
        infile = os.path.join(feat_dir, os.path.basename(path_to_video))
        return np.load(infile, allow_pickle=True)
    
    def create_features(self):
        video_dir = os.path.join(self.data_path, 'videos')
        for video in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video)
            features = self.extract_features(video_path)
            self.save_features(video_path, features)
            
if __name__ == '__main__':
    video_processor = VideoProcessor()
    video_processor.create_features()
        