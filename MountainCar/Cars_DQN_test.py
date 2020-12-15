import gym
from Cars_Env_valley import MountainCarEnv_valley
from keras.models import load_model
import numpy as np
model = load_model("car_model.h5")

env = MountainCarEnv_valley()

for i in range(100):
    state = env.reset()
    count = 0
    reward_total = 0
    while True:
        env.render()
        count += 1
        action = np.argmax(model.predict(state.reshape(-1, 2)))
        next_state, reward, done, _ = env.step(action)
        reward_total += reward
        state = next_state
        if done:
            print("Run time:", count)
            print(reward_total)
            break