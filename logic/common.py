from os import path

#########################################
# PATHS

PATH_DATA = "data"
PATH_CACHE = "cache"
PATH_OUTPUT = "output"
PATH_MODEL = path.join(PATH_CACHE, "model")

PATH_AE = path.join(PATH_MODEL, "ae.pthae")
PATH_VAE = path.join(PATH_MODEL, "vae.pthae")
PATH_VSC = path.join(PATH_MODEL, "vsc.pthae")

#########################################
# MODEL PARAMS

AE_LATENT_DIM: int = 2
VSC_LATENT_DIM: int = 2
VAE_LATENT_DIM: int = 2

DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 256
