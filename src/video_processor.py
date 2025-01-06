import os
import cv2
import numpy as np
from tensorflow.keras.application.vgg16 import VGG16
from tensorflow.keras import Model
import config

class VideoProcessor:
    def __init__(self, is_training=True):
        self.frame_count = 80
        self.annotations = self.load_annotations("data/annotations.txt")
        self.data_path = config.train_path if is_training else config.test_path
        self.model = self.load_model()

    def load_annotations(self, annotation_file):
        annotations = {}
        with open(annotation_file) as file:
            for line in file:
                video_name, annotation = line.split(' ', 1)
                annotations[video_name] = annotation
        return annotations
    
    def load_model(self):
        model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        return Model(inputs=model.input, outputs=out)
    
    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        return img
        
    def get_frames(self, path_to_video):
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
    
    def get_annotation(self, video):
        return self.annotations[video]
    
    def extract_features(self, path_to_video):
        frames = self.get_frames(path_to_video)
        samples = np.round(np.linspace(0, len(frames) - 1, self.frame_count)).astype(int)
        images = np.zeros((len(samples), 224, 224, 3))
        for i in range(len(samples)):
            image = self.load_image(frames[i])
            images[i] = image
        images = np.array(images)
        fc_features = self.model.predict(images, batch_size=128)
        image_features = np.array(fc_features)
        return image_features
    
    def save_features(self, path_to_video, features):
        feat_dir = os.path.join(self.data_path, 'feat')
        if not os.path.isdir(feat_dir):
            os.makedirs(feat_dir)

        outfile = os.path.join(feat_dir, os.path.basename(path_to_video) + '.npy')
        np.save(outfile, features)
        