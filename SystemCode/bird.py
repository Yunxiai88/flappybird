import pygame

class Bird:
    def __init__(self, x, y, image):
        self.BIRD = image
        
        self.x = x
        self.y = y
        self.tick_count = 0
        self.velocity = 0
        self.img = self.BIRD
    
    def jump(self):
        self.velocity = -6
        self.tick_count = 0
    
    def move(self):
        self.tick_count += 1

        gap = self.velocity * (self.tick_count) + 2 * self.tick_count**2
        
        if gap >= 8:
            gap = 8

        if gap < 0:
            gap -= 2
        self.y = self.y + gap
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)
    
    def draw(self, screen):
        screen.blit(self.BIRD, (self.x, self.y))
