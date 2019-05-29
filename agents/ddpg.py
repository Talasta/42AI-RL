import random
import numpy as np
from collections import deque

import tensorflow as tf
import keras.backend as K

from keras.optimizers import Adam
from keras.models import clone_model

class AgentDDPG(object):
    def __init__(
            self,
            actor_model,
            critic_model,
            compilation_dict={
                "actor_loss": "mse",
                "actor_optimizer": None,
                "critic_loss": "mse",
                "critic_optimizer": None,
            },
        ):
        # Session
        self.__dict__.update(compilation_dict)

        self.sess = tf.Session()
        K.set_session(self.sess)

        # Hyperparameters
        self.learning_rate  = 0.001
        self.epsilon        = 1.0
        self.epsilon_decay  = 0.995
        self.gamma          = 0.95

        # Experience replay
        self.memory         = deque(maxlen=2000)

        # Model settings
        self.actor_model         = actor_model
        self.target_actor_model  = clone_model(self.actor_model)
        # self.target_actor_model.build()

        self.critic_model        = critic_model
        self.target_critic_model = clone_model(self.critic_model)
        # self.target_critic_model.build()

        assert self.critic_model.output.shape[-1] == 1

        # compile models and prepare tensor pipeline for training
        self._compile_models()
        self._init_tensors_vars()

        self.sess.run(tf.initialize_all_variables())

    def _compile_models(self):
        # Check compilation dict variables
        if not hasattr(self, 'actor_loss'):
            self.actor_loss = "mse"
        if not hasattr(self, 'actor_optimizer') or self.actor_optimizer is None:
            self.actor_optimizer = Adam(lr=0.001)
        self.actor_model.compile(loss=self.actor_loss, optimizer=self.actor_optimizer)
        self.target_actor_model.compile(loss=self.actor_loss, optimizer=self.actor_optimizer)

        # Check compilation dict variables
        if not hasattr(self, 'critic_loss'):
            self.critic_loss = "mse"
        if not hasattr(self, 'critic_optimizer') or self.critic_optimizer is None:
            self.critic_optimizer = Adam(lr=0.001)
        self.critic_model.compile(loss=self.critic_loss, optimizer=self.critic_optimizer)
        self.target_critic_model.compile(loss=self.critic_loss, optimizer=self.critic_optimizer)

    def _init_tensors_vars(self):
        # Actor input / output
        self.actor_state_input = self.actor_model.input
        self.actor_output = self.actor_model.output

        # Critic inputs
        self.critic_output = self.critic_model.output
        self.critic_action_input = None
        self.critic_state_input = None

        for _tensor in self.critic_model.input:
            if self.actor_output.shape[1:] == _tensor.shape[1:]:
                self.critic_action_input = _tensor
            else:
                self.critic_state_input  = _tensor

        # Look for errors
        assert self.critic_action_input is not None
        assert self.critic_state_input  is not None
        assert self.critic_state_input.shape[1:] == self.actor_state_input.shape[1:]
        assert self.critic_output.shape[1] == 1

        # where we will feed de/dC (from critic)
        self.actor_critic_grad = tf.placeholder(
                tf.float32,
                self.actor_output.shape,
            )
        # dC/dA (from actor)
        self.actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(
                self.actor_output,
                self.actor_model_weights,
                -self.actor_critic_grad,
            )
        grads = zip(self.actor_grads, self.actor_model_weights)
        self.optimizer_actor = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        # where we calcaulte de/dC for feeding above
        self.critic_grads = tf.gradients(
                self.critic_model.output,
                self.critic_action_input,
            )

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimizer_actor, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def act(self, cur_state, random_action=None):
        self.epsilon *= self.epsilon_decay
        # Random search
        if np.random.random() < self.epsilon and random_action is not None:
            return random_action()

        return self.actor_model.predict(cur_state)

if __name__ == "__main__":

    from keras.models import Sequential, Model, clone_model
    from keras.layers import Dense, Dropout, Input
    from keras.layers import Add

    tf.logging.set_verbosity(tf.logging.ERROR)

    state_shape = (100,)
    action_shape = (5,)

    print("Build actor")
    state_input = Input(shape=state_shape)
    h1 = Dense(24, activation='relu')(state_input)
    h2 = Dense(48, activation='relu')(h1)
    h3 = Dense(24, activation='relu')(h2)
    output = Dense(action_shape[0], activation='relu')(h3)
    actor_model = Model(input=state_input, output=output)

    print("Build critic")
    state_input = Input(shape=state_shape, name="state")
    state_h1 = Dense(24, activation='relu')(state_input)
    state_h2 = Dense(48)(state_h1)
    action_input = Input(shape=action_shape, name="input")
    action_h1    = Dense(48)(action_input)
    merged    = Add()([state_h2, action_h1])
    merged_h1 = Dense(24, activation='relu')(merged)
    output = Dense(1, activation='relu')(merged_h1)
    critic_model  = Model(input=[state_input,action_input], output=output)

    ac = AgentDDPG(actor_model=actor_model, critic_model=critic_model)
