import pygame
import random
import numpy as np
import math
import struct
from network_visualizer import *

#nvcc -rdc=true -lineinfo -o runner .\biology\Genome.cpp .\Simulator.cu .\Runner.cu .\Kernels.cu .\simulations\BasicSimulation.cu .\simulations\TargetSimulation.cu .\simulations\MultibotSimulation.cu .\simulations\AirHockeySimulation.cu .\simulations\PongSimulation.cu

data_files = ["RL-bot[0]-best.data", "RL-bot[1]-best.data"]
#data_files = ["RL-bot[0]-latest.data", "RL-bot[1]-latest.data"]

numLayers = 0
# Read the all bot format
def readWeightsAndBiasesAll(data_file):
    with open(data_file, "rb") as infile:
        # Read the total number of bots
        TOTAL_BOTS = struct.unpack('i', infile.read(4))[0]

        # Read the total number of weights and neurons
        totalWeights = struct.unpack('i', infile.read(4))[0]
        totalNeurons = struct.unpack('i', infile.read(4))[0]
        print("total weights ", totalWeights)
        print("total neurons ", totalNeurons)
        # Read the number of layers and their shapes
        global numLayers
        numLayers = struct.unpack('i', infile.read(4))[0]
        layerShapes = [struct.unpack('i', infile.read(4))[0] for _ in range(numLayers)]
        print(layerShapes)
        # Allocate memory for the weights and biases
        all_weights = []
        all_biases = []

        # Read the weights and biases for each bot
        for bot in range(TOTAL_BOTS):
            # Read the weights for each layer
            weights = []
            for i in range(numLayers - 1):
                layerWeights = np.zeros(layerShapes[i] * layerShapes[i + 1], dtype=np.float32)
                for j in range(layerShapes[i] * layerShapes[i+1]):                    
                    weight = struct.unpack('f', infile.read(4))[0]
                    layerWeights[j] = weight
                weights.append(layerWeights)
                print("Layer weights : \n", layerWeights)
            #print(weights)

            # Read the biases for each layer
            biases = []
            for i in range(numLayers):
                layerBiases = np.zeros(layerShapes[i], dtype=np.float32)
                for j in range(layerShapes[i]):
                    bias = struct.unpack('f', infile.read(4))[0]
                    layerBiases[j] = bias
                biases.append(layerBiases)
                print("Layer Biases : \n", layerBiases)
            all_weights.append(weights)
            all_biases.append(biases)

    return layerShapes, all_weights, all_biases

# Initialize Pygame
pygame.init()

# Set up the game window
NETWORK_DISPLAY_WIDTH = 600

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
screen = pygame.display.set_mode((SCREEN_WIDTH + NETWORK_DISPLAY_WIDTH * 2, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Pong")

# Define paddle and ball dimensions
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
BALL_SIZE = 10

# Define paddle and ball speeds
PADDLE_SPEED = 6.5
BALL_SPEED = 8
SPEED_UP_RATE = 1.00
# Define game colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


network_display_left = pygame.Surface((NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))
network_display_right = pygame.Surface((NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))

net_displays = [network_display_left, network_display_right]
net_locations = [(0,0), (SCREEN_WIDTH + NETWORK_DISPLAY_WIDTH, 0)]

# Define game state variables
ball_x = SCREEN_WIDTH // 2
ball_y = SCREEN_HEIGHT // 2
ball_vx = random.choice([-BALL_SPEED, BALL_SPEED])
ball_vy = random.uniform(-BALL_SPEED, BALL_SPEED)
ball1 = [ball_x, ball_y, ball_vx, ball_vy]
ball2 = [ball_x, ball_y - BALL_SIZE * 2.5, -ball_vx, ball_vy]

balls = [ball1, ball2]

left_paddle_x = PADDLE_WIDTH / 2
left_paddle_y = SCREEN_HEIGHT // 2
right_paddle_x = PADDLE_WIDTH / 2 + SCREEN_WIDTH - PADDLE_WIDTH
right_paddle_y = SCREEN_HEIGHT // 2


best = 0
# Load bot networks
layershapes0, all_weights0, all_biases0 = readWeightsAndBiasesAll(data_files[0])
layershapes1, all_weights1, all_biases1 = readWeightsAndBiasesAll(data_files[1])
layershapes = layershapes0
networks = [{'weights': all_weights0[best], 'biases': all_biases0[best]},
            {'weights': all_weights1[best], 'biases': all_biases1[best]}]
#print(all_weights)


converted_all_weights0 = convert_weights(all_weights0, layershapes0)
converted_all_weights1 = convert_weights(all_weights1, layershapes1)

#print(converted_all_weights)
converted_all_weights = []

converted_all_weights.append(*converted_all_weights0)
converted_all_weights.append(*converted_all_weights1)



def forward_propagation(inputs, weights, biases, input_size, output_size, layer):
    output = np.zeros(output_size)
    #print(input_size, " ", output_size, " ", layer)
    weights = np.array(weights).reshape((input_size, output_size))

    # Initialize output to biases
    output[:] = biases

    # Compute dot product of input and weights
    output[:] += np.dot(inputs, weights)

    # Apply activation function (ReLU for non-output layers, sigmoid for output layer)
    if layer != len(layershapes) - 1:
        #leaky Relu:
        for i in range(output_size):
            if output[i] < 0:
                output[i] /= 32.0
    # else:
    #     #print('sigmoid')
    #     output[:] = 1.0 / (1.0 + np.exp(-output))
    # if layer != len(layershapes) - 1:
    #     

    
    return output

actionEffects = [PADDLE_SPEED, -PADDLE_SPEED, 0]


def get_actions_pong(state, net_weights, net_biases):
    inputs = np.array(state)
    prevLayer = inputs
    numHiddenLayers = len(layershapes) - 2
    hidden_outputs = [None] * numHiddenLayers
    #forward prop for the hidden layers
    for i in range(numHiddenLayers):
        #print("iter={}, inshape = {}, outshape = {}".format(i, layershapes[i], layershapes[i + 1]))

        hidden_outputs[i] = forward_propagation(prevLayer, net_weights[i], net_biases[i + 1], layershapes[i], layershapes[i + 1],  i)
        prevLayer = hidden_outputs[i]

    output = forward_propagation(prevLayer, net_weights[numLayers - 2], net_biases[numLayers - 1], layershapes[numLayers - 2], layershapes[numLayers - 1], numLayers - 1)
    #print(output)
    gamestate = [None] * 1
    
    chosen_action = 0
    max_value = output[0]

    for action in range(1, layershapes[-1]):
        if output[action] > max_value:
            max_value = output[action]
            chosen_action = action

    gamestate[0] = actionEffects[chosen_action]
    
    return gamestate

scores = [0,0]

# Main game loop
running = True
while running:
    screen.fill(BLACK)
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update paddle positions using neural networks
    for i in range(2):
        state = []

        # Ball 1
        if i == 0:
            state = [abs(balls[0][0] - left_paddle_x) / SCREEN_WIDTH, balls[0][1] / SCREEN_HEIGHT, balls[0][2] / BALL_SPEED, balls[0][3] / BALL_SPEED]  # Ball state
        else:
            state = [abs(balls[0][0] - right_paddle_x) / SCREEN_WIDTH, balls[0][1] / SCREEN_HEIGHT, -balls[0][2] / BALL_SPEED, balls[0][3] / BALL_SPEED]  # Ball state

        # Ball 2
        if i == 0:
            state += [abs(balls[1][0] - left_paddle_x) / SCREEN_WIDTH, balls[1][1] / SCREEN_HEIGHT, balls[1][2] / BALL_SPEED, balls[1][3] / BALL_SPEED]  # Ball state
        else:
            state += [abs(balls[1][0] - right_paddle_x) / SCREEN_WIDTH, balls[1][1] / SCREEN_HEIGHT, -balls[1][2] / BALL_SPEED, balls[1][3] / BALL_SPEED]  # Ball state

            
        if i == 0:
            state += [left_paddle_y / SCREEN_HEIGHT]  # Paddle positions
        else:
            state += [right_paddle_y / SCREEN_HEIGHT]  # Paddle positions
        
        

        #print(state)
        acceleration = get_actions_pong(state, networks[i]['weights'], networks[i]['biases'])
        
        if i == 0:
            left_paddle_y += acceleration[0]
        else:
            right_paddle_y += acceleration[0]

        mouse_pos = pygame.mouse.get_pos()
        #print(mouse_pos[0])
        if abs(mouse_pos[0] - SCREEN_WIDTH // 2 - NETWORK_DISPLAY_WIDTH) < (SCREEN_WIDTH / 2 + 20):
            
            target = []
            targety = mouse_pos[1]# - SCREEN_HEIGHT / 2
            right_paddle_y = targety

        # Keep paddles within screen boundaries
        if left_paddle_y < 0:
            left_paddle_y = 0
        elif left_paddle_y > SCREEN_HEIGHT - PADDLE_HEIGHT:
            left_paddle_y = SCREEN_HEIGHT - PADDLE_HEIGHT
        if right_paddle_y < 0:
            right_paddle_y = 0
        elif right_paddle_y > SCREEN_HEIGHT - PADDLE_HEIGHT:
            right_paddle_y = SCREEN_HEIGHT - PADDLE_HEIGHT

        #draw neural net        
        activations_left = calculate_activations(networks[i]['weights'], networks[i]['biases'], state, layershapes, numLayers)
        display_activations(activations_left, converted_all_weights[i], net_displays[i])
        #display_activations2(activations_left, converted_all_weights[i], net_displays[i], SCREEN_HEIGHT)
        screen.blit(net_displays[i], net_locations[i])

    # update game state
    for ball in range (2):
            
        balls[ball][0] += balls[ball][2]
        balls[ball][1] += balls[ball][3]

        if balls[ball][0] - BALL_SIZE <= left_paddle_x + PADDLE_WIDTH and balls[ball][1] + BALL_SIZE >= left_paddle_y - PADDLE_HEIGHT / 2 and balls[ball][1] - BALL_SIZE <= left_paddle_y + PADDLE_HEIGHT / 2 and balls[ball][2] < 0:
            balls[ball][2] = -balls[ball][2] * SPEED_UP_RATE
            #balls[ball][3] += (balls[ball][1] - left_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
            balls[ball][3] = (random.random() - 0.5) * BALL_SPEED * 2
            balls[ball][0] += balls[ball][2]
            balls[ball][1] += balls[ball][3]

        if balls[ball][0] + BALL_SIZE >= right_paddle_x and balls[ball][1] + BALL_SIZE >= right_paddle_y  - PADDLE_HEIGHT / 2 and balls[ball][1] - BALL_SIZE <= right_paddle_y + PADDLE_HEIGHT / 2 and balls[ball][2] > 0:
            balls[ball][2] = -balls[ball][2] * SPEED_UP_RATE
            #balls[ball][3] += (balls[ball][1] - right_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
            balls[ball][3] = (random.random() - 0.5) * BALL_SPEED * 2

            balls[ball][0] += balls[ball][2]
            balls[ball][1] += balls[ball][3]
        if balls[ball][1] - BALL_SIZE < 0 or balls[ball][1] + BALL_SIZE > SCREEN_HEIGHT:
            balls[ball][3] = -balls[ball][3]
        
        # Score
        if balls[ball][0] < 0 or balls[ball][0] > SCREEN_WIDTH:
            if balls[ball][0] < 0:
                scores[1] += 1
            else:
                scores[0] += 1

            balls[ball][2] = -balls[ball][2]
            balls[ball][3] /= 2

            balls[ball][0] += balls[ball][2] * 2
            #balls[ball][1] = SCREEN_HEIGHT // 2

            #right_paddle_y = SCREEN_HEIGHT // 2
            #left_paddle_y = SCREEN_HEIGHT // 2

        # draw game objects
        # TESTING Y OFFSET DRAWING
        pygame.draw.rect(screen, WHITE, (left_paddle_x + NETWORK_DISPLAY_WIDTH, left_paddle_y - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, WHITE, (right_paddle_x + NETWORK_DISPLAY_WIDTH, right_paddle_y  - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        if ball == 0:
            pygame.draw.circle(screen, (255,255,0), (int(balls[ball][0]) + NETWORK_DISPLAY_WIDTH, int(balls[ball][1])), BALL_SIZE)
        else:
            pygame.draw.circle(screen, (0,255,255), (int(balls[ball][0]) + NETWORK_DISPLAY_WIDTH, int(balls[ball][1])), BALL_SIZE)

        #DRAW THE PLAYER SCORES
        # Draw the scores
        font = pygame.font.Font(None, 36)
        score1_text = font.render("" + str(scores[0]), True, WHITE)
        score2_text = font.render("" + str(scores[1]), True, WHITE)
        score1_rect = score1_text.get_rect()
        score2_rect = score2_text.get_rect()
        spacing = 20
        score1_rect.midtop = (SCREEN_WIDTH // 2 - spacing + NETWORK_DISPLAY_WIDTH, 10)
        score2_rect.midtop = (SCREEN_WIDTH // 2 + spacing + NETWORK_DISPLAY_WIDTH, 10)
        screen.blit(score1_text, score1_rect)
        screen.blit(score2_text, score2_rect)
        pygame.display.flip()
    # update the display
    pygame.display.update()
    clock.tick(72)

# quit Pygamew
pygame.quit()