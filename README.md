# Selective_Noise_Cancellation
A project derived from the team project of the course AAI3201 Deep Learning: Theory and Practice offered in the 2022 Fall Semester at Yonsei University.  

## Encoder-Decoder
### data_preprocessing.py 

This code shows how we converted our input data for the encoder-decoder model:m4a -> wav -> split into 10s -> spectrogram.
When input data(field recordings(.m4a) in subway environment, field recordings(.m4a) other than subway environment)
passes through this code, it produces corresponding spectrograms(.png) of size 128x431.

### getDataLoader.py
This code gets saved spectrogram image from data_preprocessing.py as input to produce (image,label) dataset in batches.

### Autoencoder.py
Main part of autoencoder model. This code shows the architecture of our autoencoder model.

### latent_Space_Visualization.py
This file imports getDataLoader.py and Autoencoder.py internally.
By importing two files this code could stand alone as a latent space visualizer of this autoencoder model.

### Classifier.py
This file imports getDataLoader.py and Autoencoder.py.
This lets Classifier.py get latent space vectors of the autoencoder as an input and map that into internal mlp layers to produce output;
whether the environment of given input data(to the autoencoder) is subway or not.
Since it has imported all necessary .py files, it could stand alone as classifier.

## Sound Separation

