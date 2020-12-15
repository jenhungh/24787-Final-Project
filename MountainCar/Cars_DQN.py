import keras
import random
from collections import deque
import gym
from Cars_Env_valley import MountainCarEnv_valley
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD


class Agent():
    def __init__(self, action_set, observation_space):
        """
        Initialize
        :param action_set
        :param observation_space: shape of states
        """
        # Discount factors
        self.gamma = 1.0
        # batch size
        self.batch_size = 50
        # memory list
        self.memory = deque(maxlen=2000000)
        # exploraiton rate: greedy
        self.greedy = 1.0
        # action sets
        self.action_set = act ㄐㄟˉion_set
        # states
        self.observation_space = observation_space
        # NN model
        self.model = self.init_netWork()

    # NN model
    def init_netWork(self):
        """
        Build the model 
        :return: model
        """
        model = Sequential()
        model.add(Dense(64 * 4, activation="tanh", input_dim=self.observation_space.shape[0]))
        model.add(Dense(64 * 4, activation="tanh"))
        model.add(Dense(self.action_set.n, activation="linear"))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(lr=0.001))
        return model
    
    # Create memory list
    def add_memory(self, sample):
        self.memory.append(sample)

    # Update greedy
    def update_greedy(self):
        if self.greedy > 0.01:
            self.greedy *= 0.995

    # Train the NN model
    def train_model(self):
        """
        Train the model 
        """
        # Select the training set from the memory list
        train_sample = random.sample(self.memory, k=self.batch_size)

        # Get current state and next state
        train_states = []
        next_states = []
        for sample in train_sample:
            cur_state, action, r, next_state, done = sample
            next_states.append(next_state)
            train_states.append(cur_state)
        next_states = np.array(next_states)
        train_states = np.array(train_states)
        
        # Predict the q of the next state
        next_states_q = self.model.predict(next_states)

        # Predict the q of the training set 
        state_q = self.model.predict_on_batch(train_states)

        # Compute Q reality
        for index, sample in enumerate(train_sample):
            cur_state, action, r, next_state, done = sample
            if not done:
                state_q[index][action] = r + self.gamma * np.max(next_states_q[index])
            else:
                state_q[index][action] = r
        
        # Train the NN model
        self.model.train_on_batch(train_states, state_q)

    # Run the environment
    def act(self, env, action):
        """
        Run the environment
        :param env: import the environment
        :param action: run the actions
        :return: next_state, reward, done
        """
        next_state, reward, done, _ = env.step(action)

        # Change the rewards to let the training process converges faster
        if done:
            if reward < 0:
                reward = -100
            else:
                reward = 10
        else:
            if next_state[0] >= 0.4:
                reward += 1

        return next_state, reward, done
    
    # Get the best actions
    def get_best_action(self, state):
        # Exploration
        if random.random() < self.greedy:
            return self.action_set.sample()
        # Exploitation
        else:
            return np.argmax(self.model.predict(state.reshape(-1, 2)))

# Main funciton
if __name__ == "__main__":
    # set the training episodes
    episodes = 10000
    # import the environment
    env = MountainCarEnv_valley() 
    # import the agent
    agent = Agent(env.action_space, env.observation_space)
    # how many times we run act()
    counts = deque(maxlen=10)

    for episode in range(episodes):
        count = 0
        # reset the environment
        state = env.reset()

        # explore the environment after first 5 episodes 
        if episode >= 5:
            agent.update_greedy()

        while True:
            count += 1
            # Get the best action
            action = agent.get_best_action(state)
            next_state, reward, done = agent.act(env, action)
            agent.add_memory((state, action, reward, next_state, done))
            # Fill the memory list in the first 5 episodes
            if episode >= 5:
                agent.train_model()
            state = next_state
            if done:
                # fill in counts
                counts.append(count)
                print(f"In {episode + 1} episode，agent runs {count} times.")

                # set the tolerance to break
                if len(counts) == 10 and np.mean(counts) < 160:
                    agent.model.save("car_model.h5")
                    exit(0)
                break