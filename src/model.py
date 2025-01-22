import os
import joblib
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model, load_model
import config


def create_models():
    # Encoder
    encoder_inputs = Input(shape=(config.time_steps_encoder, config.num_encoder_tokens), name='encoder_inputs')
    encoder_lstm = LSTM(config.latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None, config.num_decoder_tokens), name='decoder_inputs')
    decoder_lstm = LSTM(config.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(config.num_decoder_tokens, activation='softmax', name='decoder_relu')

    # Training Model (combined)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_dense(decoder_outputs)
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Encoder Model (for inference)
    encoder_inference_model = Model(encoder_inputs, encoder_states)

    # Decoder Model (for inference)
    decoder_inference_model = create_decoder_inference_model(decoder_dense, decoder_inputs, decoder_lstm)

    return training_model, encoder_inference_model, decoder_inference_model


def create_decoder_inference_model(decoder_dense, decoder_inputs, decoder_lstm):
    """
    Create the decoder inference model used for prediction.
    """
    decoder_state_input_h = Input(shape=(config.latent_dim,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(config.latent_dim,), name='decoder_state_input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # Reuse LSTM and Dense layers
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return decoder_model


def load_inference_models():
    """
    Load models for inference: encoder, decoder, and tokenizer.
    """
    # Load the tokenizer
    with open(os.path.join(config.save_model_path, 'tokenizer' + str(config.num_decoder_tokens)), 'rb') as file:
        tokenizer = joblib.load(file)

    # Load the encoder model
    inf_encoder_model = load_model(os.path.join(config.save_model_path, 'encoder_model.h5'))

    # Load the decoder model
    decoder_inputs = Input(shape=(None, config.num_decoder_tokens))
    decoder_dense = Dense(config.num_decoder_tokens, activation='softmax')
    decoder_lstm = LSTM(config.latent_dim, return_sequences=True, return_state=True)
    inf_decoder_model = create_decoder_inference_model(decoder_dense, decoder_inputs, decoder_lstm)
    inf_decoder_model.load_weights(os.path.join(config.save_model_path, 'decoder_model_weights.h5'))

    return tokenizer, inf_encoder_model, inf_decoder_model
