"""
solving pendulum using actor-critic model
"""

import gym
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply

from agents.ddpg import AgentDDPG

def model_actor(state_shape, action_shape):

    state_input  = Input(shape=state_shape)
    h1           = Dense(24, activation='relu')(state_input)
    h2           = Dense(48, activation='relu')(h1)
    h3           = Dense(24, activation='relu')(h2)

    output       = Dense(action_shape[0], activation='relu')(h3)

    return Model(input=state_input, output=output)

def model_critic(state_shape, action_shape):

    state_input  = Input(shape=state_shape, name="state")
    state_h1     = Dense(24, activation='relu')(state_input)
    state_h2     = Dense(48)(state_h1)

    action_input = Input(shape=action_shape, name="input")
    action_h1    = Dense(48)(action_input)

    merged       = Add()([state_h2, action_h1])
    merged_h1    = Dense(24, activation='relu')(merged)

    output       = Dense(1, activation='relu')(merged_h1)

    return Model(input=[state_input,action_input], output=output)

def main_pendulum():
    tf.logging.set_verbosity(tf.logging.ERROR)

    env = gym.make("Pendulum-v0")

    state_shape  = env.observation_space.shape
    action_shape = env.action_space.shape

    actor_model  = model_actor(state_shape, action_shape)
    critic_model = model_critic(state_shape, action_shape)

    actor_critic = AgentDDPG(actor_model=actor_model, critic_model=critic_model)

    num_trials = 10000
    trial_len  = 500

    cur_state = env.reset()
    action = env.action_space.sample()

    while True:
        env.render()
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        action = actor_critic.act(cur_state, random_action=env.action_space.sample)
        action = action.reshape((1, env.action_space.shape[0]))

        new_state, reward, done, _ = env.step(action)
        print("State[ {:5.2f}, {:5.2f}, {:5.2f} ] :: Action[ {:20.2f} ] :: Reward[ {:5.2f} ]".format(cur_state[0][0], cur_state[0][1], cur_state[0][2], action[0][0], reward[0]), end="\r")
        new_state = new_state.reshape((1, env.observation_space.shape[0]))

        actor_critic.remember(cur_state, action, reward, new_state, done)
        actor_critic.train()

        cur_state = new_state

if __name__ == "__main__":
    main_pendulum()
