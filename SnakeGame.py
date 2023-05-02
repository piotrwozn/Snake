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
        self.reset()

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

        new_head = (self.snake[-1][0]+new_direction[0], self.snake[-1][1]+new_direction[1])

        if new_head in self.snake[:-1] or new_head[0] == 0 or new_head[1] == 0 or new_head[0] == self.screen_size//self.grid_size-1 or new_head[1] == self.screen_size//self.grid_size-1:
            return self.get_state(), -100, True

        self.snake.append(new_head)
        self.direction = new_direction

        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
        else:
            self.snake.pop(0)

        return self.get_state(), 10, False

    def is_near_wall(self):
        head = self.snake[-1]
        if head[0] == 1 or head[1] == 1 or head[0] == self.screen_size // self.grid_size - 2 or head[
            1] == self.screen_size // self.grid_size - 2:
            return 1
        return 0

    def get_state(self):
        snake_segments = [segment for part in self.snake[:-1] for segment in part]
        padding = [0] * (24 - len(snake_segments))
        return [
            self.direction[0], self.direction[1],
            self.food[0], self.food[1],
            self.snake[-1][0], self.snake[-1][1]
        ] + snake_segments + padding + [self.is_near_wall()]


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
    return np.reshape(state[:13], [1, 13])

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
    agent = DQNAgent()
    batch_size = 128

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
            pygame.time.wait(100)
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