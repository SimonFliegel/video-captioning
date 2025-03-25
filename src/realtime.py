from inference import VideoCaptionInference
import os
import mimetypes

import config

def predict_realtime(directory):
    """
    Predict captions for videos in real-time.
    :param directory: the directory containing the videos.
    """
    inference = VideoCaptionInference(config)
    videos = os.listdir(directory)
    for i in range(len(videos)):
        video_path = os.path.join(directory, videos[i])
        mime_type = mimetypes.guess_type(video_path)
        if mime_type[0] is None or not mime_type[0].startswith('video'):
            print("Skipping file: ", video_path)
            continue
        print("Predicting for video: ", videos[i])
        inference.predict_realtime(video_path)
        if i < len(videos) - 1:
            print("Do you want to continue with the next video? (y/n)")
            answer = input()
            if answer not in ['y', 'Y']:
                break


if __name__ == '__main__':
    predict_realtime(config.realtime_path)