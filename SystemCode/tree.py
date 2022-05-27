import pygame
import random

WIDTH = 60
HEIGHT = 300

class Tree:
    def __init__(self, x, image):
        self.TREE = image
        
        self.BOTTOM_TREE = self.TREE
        self.TOP_TREE = pygame.transform.flip(self.TREE, False, True)

        self.gap = 200
        self.velocity = 5

        self.x = x
        self.y = random.randrange(150, HEIGHT)

        self.top = self.y - self.TOP_TREE.get_height()
        self.bottom = self.y + self.gap

        self.passed = False

    def move(self):
        self.x -= self.velocity
    
    def isCollide(self, bird):
        bird_mask = bird.get_mask()

        top_mask = pygame.mask.from_surface(self.TOP_TREE)
        bottom_mask = pygame.mask.from_surface(self.BOTTOM_TREE)
        
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True
        
        return False
    
    def draw(self, screen):
        screen.blit(self.TOP_TREE, (self.x, self.top))
        screen.blit(self.BOTTOM_TREE, (self.x, self.bottom))
