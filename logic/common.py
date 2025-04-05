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

PATH_VISUALIZATION = path.join(PATH_OUTPUT, "graphics")

#########################################
# MODEL PARAMS

AE_LATENT_DIM: int = 2
VSC_LATENT_DIM: int = 2
VAE_LATENT_DIM: int = 2

DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LEARNING_RATE: float = 1e-3

DEFAULT_VSC_N_WARMUP: int = 100
DEFAULT_VSC_DELTA_LAMBDA: float = 0.01
DEFAULT_VSC_L: int = 1

#########################################
# DATASET PARAMS

MNIST_IMAGE_WIDTH = 28
MNIST_FASHION_WIDTH = 28
