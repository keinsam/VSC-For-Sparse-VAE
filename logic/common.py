from os import path

#########################################
# PATHS

PATH_DATA = "data"
PATH_CACHE = "cache"
PATH_OUTPUT = "output"
PATH_MODEL = path.join(PATH_CACHE, "model")

MODEL_EXTENSION = ".pthae"
PATH_AE = path.join(PATH_MODEL, "ae" + MODEL_EXTENSION)
PATH_VAE = path.join(PATH_MODEL, "vae" + MODEL_EXTENSION)
PATH_VSC = path.join(PATH_MODEL, "vsc" + MODEL_EXTENSION)
PATH_VSC_WARMUP = path.join(PATH_MODEL, "vsc_warmup" + MODEL_EXTENSION)

#########################################
# MODEL PARAMS

AE_LATENT_DIM: int = 2
VSC_LATENT_DIM: int = 2
VAE_LATENT_DIM: int = 2

DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 256

#########################################
# DATASET PARAMS

MNIST_IMAGE_WIDTH = 28
MNIST_FASHION_WIDTH = 28
