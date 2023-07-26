import pygame
import sys
from random import random
import numpy as np
from NN import NeuralNetwork
global grid, width, height, tile_size, radius, data
width, height = 800, 600
tile_size = min(width, height) // 28
radius = 2
data = [(i, 0) for i in range(10)]
if 0:
    grid = [[random() for __ in range(28)] for _ in range(28)]
else:
    grid = [[0 for __ in range(28)] for _ in range(28)]

global model
model = None

def displayData(screen):
    global data
    font_size = height // 25
    font = pygame.font.SysFont('Futura', font_size)
    for i in range(len(data)):
        text = font.render(f'{data[i][0]} - {data[i][1] * 100:.2f}%', True, (255, 255, 255))
        screen.blit(text, (10, 10 + i * font_size * 2))
    font = pygame.font.SysFont('Futura', 15)
    screen.blit(font.render(f'radius - {radius}', True, (255, 255, 255)), (10, height - 10 - 60))
    screen.blit(font.render('LMB - draw', True, (255, 255, 255)), (10, height - 10 - 45))
    screen.blit(font.render('RMB - erase', True, (255, 255, 255)), (10, height - 10 - 30))
    screen.blit(font.render('c - clear', True, (255, 255, 255)), (10, height - 10 - 15))

def draw(screen):    
    screen.fill((35, 35, 35))
    displayData(screen)
    for i in range(28):
        for j in range(28):
            color = (255 * grid[i][j], 255 * grid[i][j], 255 * grid[i][j])
            pygame.draw.rect(screen, color, (i * tile_size + width // 4, j * tile_size + height // 100, tile_size, tile_size))

def paint(pos, button):
    global grid, radius
    x, y = pos
    mult = .5
    x = (x - width // 4) // tile_size
    y = (y - height // 100) // tile_size
    if 0 <= x < 28 and 0 <= y < 28:
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if not (0 <= x + j < 28) or not (0 <= y + i < 28) or (i ** 2 + j ** 2 > radius ** 2):
                        continue
                    if button == 1:
                        try:
                            grid[x + j][y + i] += mult * (1 / (1 + i ** 2 + j ** 2))
                            grid[x + j][y + i] = min(1, grid[x + j][y + i])
                        except:
                            pass
                    elif button == 3:
                        try:
                            grid[x + j][y + i] -= mult
                            grid[x + j][y + i] = max(0, grid[x + j][y + i])
                        except:
                            pass

def main():
    global grid, radius, data
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("digit guess")
    pygame.display.set_icon(pygame.image.load('./logo.png'))
    drag = False
    button = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and button != event.button:
                drag = True
                button = event.button
            elif event.type == pygame.MOUSEBUTTONUP:
                drag = False
                button = 0
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                grid = [[0 for __ in range(28)] for _ in range(28)]
            if event.type == pygame.MOUSEWHEEL:
                if event.y < 0:
                    radius += 1
                elif event.y > 0:
                    radius -= 1
                    radius = max(1, radius)
            if drag:
                paint(pygame.mouse.get_pos(), button)

        data = predict().tolist()[0]
        data = [(i, data[i]) for i in range(10)]
        data.sort(key = lambda x: x[1], reverse=True)
        draw(screen)

        pygame.display.update()

def predict():
    global model, grid
    return model.predict_proba(np.array(list(list(x) for x in zip(*grid))[::-1]).reshape((1, 784)) * 255)

def loadModel():
    from joblib import load
    global model
    model = load('v2.joblib').nn

if __name__ == '__main__':
    loadModel()
    main()