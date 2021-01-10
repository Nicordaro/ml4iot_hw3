import os
import numpy as np
import tensorflow as tf

def read(audio_binary, sampling_rate):
    audio,_ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)  
    audio = pad(audio, sampling_rate)
        
    return audio

def pad(audio, sampling_rate):
    zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([sampling_rate])
        
    return audio

def preprocess(audio_binary, sampling_rate, frame_length, frame_step, num_mel_bins=None, lower_frequency=None,
                upper_frequency=None, num_coefficients=None, mfcc=False):
    
    audio = read(audio_binary, sampling_rate)

    stft = tf.signal.stft(audio, frame_length=frame_length,
                          frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    if mfcc == False:
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [55,55])
        spectrogram = tf.expand_dims(spectrogram, 0)
        return spectrogram
    
    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                        sampling_rate, lower_frequency,
                                                                        upper_frequency)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients]
    mfccs = tf.expand_dims(mfccs, -1)
    mfccs = tf.expand_dims(mfccs, 0)
    return mfccs