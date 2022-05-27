import pygame

class BackGround:
    def __init__(self, image):
        self.BACKGROUND = image
        
        self.x = 0
        self.y = 0
        
    def draw(self, screen):
        screen.blit(self.BACKGROUND, (self.x, self.y))