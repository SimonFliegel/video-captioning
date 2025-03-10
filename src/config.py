# Description: Configuration file for the model containing global variables.
train_path = "../data/train"
test_path = "../data/test"
realtime_path = "../data/realtime"
save_model_path = "../models/model_default"
batch_size = 320
learning_rate = 0.0007
epochs = 150
latent_dim = 512 # number of hidden units in LSTM
num_encoder_tokens = 4096 # VGG16 features
num_decoder_tokens = 1500 # vocab size
time_steps_encoder = 80 # frames
time_steps_decoder = 10 # words
max_probability = -1
validation_split = 0.15
min_length = 6 # words
max_length = 10 # words
search_type = "greedy"
