import random
import sys
import pygame
from pygame.locals import *

from models import ReplayMemory
from network import DeepNetwork
from deepnetwork import DeepNetwork1

# --------------     set configration ---------------------
pygame.display.set_caption('Flappy Bird')

if __name__ == "__main__":
    # initial experience pool
    buffer = ReplayMemory()

    # build network
    #model = DeepNetwork1()
    #model.load_model()

    model = DeepNetwork(model_path='saved_networks')

    # train the network
    model.train_model(buffer)