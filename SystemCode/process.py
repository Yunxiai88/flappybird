import pygame
import numpy as np

from tree import Tree
from bird import Bird

fps = 30
WIN_WIDTH = 690
WIN_HEIGHT = 600

pygame.font.init()
fps_clock = pygame.time.Clock()

SCORE_FONT = pygame.font.SysFont("comicsans", 20)
SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

def processTree(trees, bird, score):
    terminal = False
    add_tree = False
    remove_trees = []

    reward = 0.1

    for tree in trees:
        tree.move()

        if tree.isCollide(bird):
            reward = -1
            terminal = True

        if tree.x + tree.TOP_TREE.get_width() < 0:
            remove_trees.append(tree)
    
        if not tree.passed and tree.x < bird.x:
            tree.passed = True
            add_tree = True
        
    if add_tree:
        score += 1
        reward = 1
        trees.append(Tree(WIN_WIDTH))

    for rmt in remove_trees:
        trees.remove(rmt)
    
    return trees, score, reward, terminal

def processBird(bird, trees, action):
    ind = 0

    if len(trees) > 1 and bird.x > trees[0].x + trees[0].TOP_TREE.get_width():  # determine whether to use the first or second
        ind = 1

    # bird flying
    bird.move()

    if action[0] == 0:
        print('jump..')
        bird.jump()
    
    status = [(bird.y, abs(bird.y - trees[ind].y), abs(bird.y - trees[ind].bottom))]

    return status

def draw_screen(ground, trees, bird, score):
    # draw backgroud
    BG_IMAGE = pygame.transform.scale(pygame.image.load("img/bg.png").convert_alpha(), (WIN_WIDTH, 900))
    SCREEN.blit(BG_IMAGE, (0,0))

    # draw ground
    ground.draw(SCREEN)

    # draw trees
    for tree in trees:
        tree.draw(SCREEN)
    
    # draw birdss
    bird.draw(SCREEN)
    
    # draw score
    score_text = SCORE_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    SCREEN.blit(score_text, (WIN_WIDTH - score_text.get_width() - 15, 5))

    # generate image
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())

    pygame.display.update()

    fps_clock.tick(fps)

    return image_data, score