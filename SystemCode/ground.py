import pygame

class Ground:
    def __init__(self, image):
        self.GROUND = image
        self.WIDTH  = self.GROUND.get_width()

        self.x1 = 0
        self.x2 = self.WIDTH
        self.y = 600 - 60

        self.velocity = 5
    
    def move(self):
        self.x1 -= self.velocity
        self.x2 -= self.velocity

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
        
    def draw(self, screen):
        screen.blit(self.GROUND, (self.x1, self.y))
        screen.blit(self.GROUND, (self.x2, self.y))