import pygame

from tree import Tree
from bird import Bird
from ground import Ground
from background import BackGround

FPS = 30
FLOOR = 530
WIN_WIDTH = 600
WIN_HEIGHT = 600

DRAW_LINE = False

pygame.font.init()
FPSCLOCK = pygame.time.Clock()

SCORE_FONT = pygame.font.SysFont("comicsans", 20)
SCREEN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

BIRD = pygame.transform.scale(pygame.image.load("img/bird.png").convert_alpha(), (40, 30))
TREE = pygame.transform.scale(pygame.image.load("img/pipe.png").convert_alpha(), (60, 400))
GROUND = pygame.transform.scale(pygame.image.load("img/ground.png").convert_alpha(), (600, 60))
BACKGROUND = pygame.transform.scale(pygame.image.load("img/background.png").convert_alpha(), (WIN_WIDTH, WIN_HEIGHT))


class GameState:
    def __init__(self):
        self.score = 0
        self.bird = Bird(250, 250, BIRD)
        self.trees = [Tree(600, TREE)]
        
        self.ground = Ground(GROUND)
        self.background = BackGround(BACKGROUND)
    
    def draw_screen(self):
        # draw backgroud
        self.background.draw(SCREEN)

        # draw trees
        for tree in self.trees:
            tree.draw(SCREEN)
        
        # draw ground
        self.ground.draw(SCREEN)
        
        # draw birds
        self.bird.draw(SCREEN)

        pipe_ind = 0
        if len(self.trees) > 1 and self.bird.x > self.trees[0].x + self.trees[0].TOP_TREE.get_width():
            pipe_ind = 1
        
        if DRAW_LINE:
            try:
                pygame.draw.line(SCREEN, (255,0,0), (self.bird.x+self.bird.img.get_width()/2, self.bird.y + self.bird.img.get_height()/2), (self.trees[pipe_ind].x + self.trees[pipe_ind].TOP_TREE.get_width()/2, self.trees[pipe_ind].y), 5)
                pygame.draw.line(SCREEN, (255,0,0), (self.bird.x+self.bird.img.get_width()/2, self.bird.y + self.bird.img.get_height()/2), (self.trees[pipe_ind].x + self.trees[pipe_ind].BOTTOM_TREE.get_width()/2, self.trees[pipe_ind].bottom), 5)
            except:
                pass
        
        # draw score
        score_text = SCORE_FONT.render("Score: " + str(self.score), 1, (255, 255, 255))
        SCREEN.blit(score_text, (WIN_WIDTH - score_text.get_width() - 15, 5))

        # generate image
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        pygame.display.update()

        FPSCLOCK.tick(FPS)

        return image_data
    
    def play(self, action):
        pygame.event.pump()
        
        reward = 0.1
        terminal = False

        # 1. process trees
        remove_trees = []

        for tree in self.trees:
            tree.move()

            if tree.isCollide(self.bird):
                reward = -100
                terminal = True

            # remove tree if out of window
            if tree.x + tree.TOP_TREE.get_width() < 0:
                remove_trees.append(tree)
            
            # give reward if bird pass the tree
            if not tree.passed and tree.x < self.bird.x:
                tree.passed = True

                reward = 1
                self.score += 1
        
        # add a tree if last tree exceed the defined distance
        if WIN_WIDTH - self.trees[-1].x == 250:
            self.trees.append(Tree(WIN_WIDTH, TREE))

        for rmt in remove_trees:
            self.trees.remove(rmt)
        
        # 2. moving ground
        self.ground.move()

        # 3. bird flying
        if action[1] == 1:
            self.bird.jump()
        
        self.bird.move()
        
        if self.bird.y + self.bird.img.get_height() >= FLOOR or self.bird.y < 5:
            reward = -100
            terminal = True

        # draw screen
        image_data = self.draw_screen()

        return image_data, reward, terminal