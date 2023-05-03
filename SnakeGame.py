import time

import pygame
import numpy as np
import random
from DQNAgent import DQNAgent
import os
import glob
import re


class SnakeGame:
    def __init__(self, screen_size=400, grid_size=20):
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.snake = [(screen_size // (2 * grid_size) * grid_size, screen_size // (2 * grid_size) * grid_size)]
        self.direction = (grid_size, 0)
        self.food = self.generate_food()
        self.score = 0
        self.previous_food_distance = np.linalg.norm(np.array(self.snake[-1]) - np.array(self.food))
    def reset(self):
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption('Snake DQN')
        self.snake = [(2, 2), (2, 3), (2, 4)]
        self.food = self.generate_food()
        self.direction = (0, -1)
        self.score = 0
        return self.get_state()

    def generate_food(self):
        food = (random.randint(1, self.screen_size//self.grid_size-2), random.randint(1, self.screen_size//self.grid_size-2))
        while food in self.snake:
            food = (random.randint(1, self.screen_size//self.grid_size-2), random.randint(1, self.screen_size//self.grid_size-2))
        return food

    def step(self, action):
        if action == 0:
            new_direction = (-self.direction[1], self.direction[0])
        elif action == 1:
            new_direction = self.direction
        else:
            new_direction = (self.direction[1], -self.direction[0])

        new_head = (self.snake[-1][0] + new_direction[0], self.snake[-1][1] + new_direction[1])

        if new_head in self.snake[:-1] or new_head[0] == 0 or new_head[1] == 0 or new_head[
            0] == self.screen_size // self.grid_size - 1 or new_head[1] == self.screen_size // self.grid_size - 1:
            return self.get_state(), -50, True  # Decreased penalty for hitting the wall

        self.snake.append(new_head)
        self.direction = new_direction

        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
            reward = 100  # Increased reward for collecting food
        else:
            self.snake.pop(0)
            reward = 0

        # Reward for getting closer to the food
        food_distance = np.linalg.norm(np.array(new_head) - np.array(self.food))
        reward_distance = 10 * (self.previous_food_distance - food_distance)
        if food_distance < self.previous_food_distance:
            reward += reward_distance
        else:
            reward -= reward_distance

        self.previous_food_distance = food_distance

        return self.get_state(), reward, False

    def is_near_wall(self):
        head = self.snake[-1]
        if head[0] == 1 or head[1] == 1 or head[0] == self.screen_size // self.grid_size - 2 or head[
            1] == self.screen_size // self.grid_size - 2:
            return 1
        return 0

    def get_state(self):
        state = [[0 for _ in range(self.screen_size // self.grid_size)] for _ in
                 range(self.screen_size // self.grid_size)]

        for segment in self.snake:
            state[segment[1]][segment[0]] = 1

        state[self.food[1]][self.food[0]] = 2

        for i in range(self.screen_size // self.grid_size):
            state[0][i] = 3
            state[i][0] = 3
            state[self.screen_size // self.grid_size - 1][i] = 3
            state[i][self.screen_size // self.grid_size - 1] = 3

        return state

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, 0, self.screen_size, self.grid_size))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, 0, self.grid_size, self.screen_size))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, self.screen_size-self.grid_size, self.screen_size, self.grid_size))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(self.screen_size-self.grid_size, 0, self.grid_size, self.screen_size))

        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                             pygame.Rect(segment[0] * self.grid_size, segment[1] * self.grid_size, self.grid_size,
                                         self.grid_size))

        pygame.draw.rect(self.screen, (255, 0, 0),
                         pygame.Rect(self.food[0] * self.grid_size, self.food[1] * self.grid_size, self.grid_size,
                                     self.grid_size))

        myfont = pygame.font.SysFont("monospace", 15)
        label = myfont.render("Score: {}".format(self.score), 1, (255, 255, 0))
        self.screen.blit(label, (self.screen_size // 2, 10))

        pygame.display.flip()


def preprocess_state(state):
    return np.reshape(state, (1, -1))


def latest_weights_file():
    list_of_files = glob.glob('*.h5')
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def extract_episode_number(file_name):
    match = re.search(r'snake_dqn_weights_(\d+).h5', file_name)
    if match:
        return int(match.group(1))
    return None

def delete_weights_file(file_name):
    try:
        os.remove(file_name)
    except OSError as e:
        print(f"Error deleting file {file_name}: {e.strerror}")

def main():
    pygame.init()
    game = SnakeGame()
    agent = DQNAgent(game)
    batch_size = 32

    latest_file = latest_weights_file()
    if latest_file is not None:
        print("Loading weights from", latest_file)
        agent.load(latest_file)
        episode_number = extract_episode_number(latest_file)
        if episode_number is not None:
            episode = episode_number + 1
        else:
            episode = 1
    else:
        episode = 1

    while True:
        state = game.reset()
        state = preprocess_state(state)
        done = False
        while not done:
            game.render()
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                with open("progress_log.txt", "a") as progress_log:
                    progress_log.write("Episode: {}, Score: {}\n".format(episode, game.score))
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if game.score >= 90:
            print("Snake has won the game!")
            break

        # Save weights and remove previous weights file
        new_weights_file = f"snake_dqn_weights_{episode}.h5"
        agent.save(new_weights_file)
        if latest_file is not None:
            delete_weights_file(latest_file)
        latest_file = new_weights_file
        episode += 1

if __name__ == "__main__":
    main()