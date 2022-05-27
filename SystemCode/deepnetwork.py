import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from keras import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten

from games import GameState
from models import ReplayMemory

OBSERVE = 1000.
EXPLORE = 2000000.   
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
ACTIONS = 2
BATCH = 32

class DeepNetwork1:
    def __init__(self, lr=1.0e-6, gamma=0.99, train=True):
        self.gamma = gamma
        self.tau = 0.01
        self.time_step = 0
        self.epsilon_step = 0
        self.learning_rate = lr
        self.epsilon = INITIAL_EPSILON
        
        # state input
        self.state_input  = Input(shape=(80, 80, 4), dtype=tf.dtypes.float32)
        self.state_input_ = Input(shape=(80, 80, 4), dtype=tf.dtypes.float32)

        # action input
        self.action  = Input(shape=(2), dtype=tf.dtypes.float32)
        self.action_ = Input(shape=(2), dtype=tf.dtypes.float32)
        
        # Q value layer
        self.Q = self.create_model(self.state_input, trainable=True)
        self.Q_= self.create_model(self.state_input_, trainable=False)
    
    def create_model(self, input, trainable):
        conv1 = Conv2D(32, (8,8), strides=4, activation='relu', padding='same', trainable=trainable)(input)
        max1  = MaxPooling2D(pool_size=(2,2), strides=2)(conv1)
        conv2 = Conv2D(64, (4,4), strides=1, activation='relu', padding='same', trainable=trainable)(max1)
        conv3 = Conv2D(64, (3,3), strides=1, activation='relu', padding='same', trainable=trainable)(conv2)
        
        flat1 = Flatten()(conv3)
        dens1 = Dense(512, activation='relu', trainable=trainable)(flat1)
        dens2 = Dense(2, trainable=trainable)(dens1)

        model = Model(inputs=input, outputs=dens2)
        
        model.compile(loss='mse',optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()

        return model
    
    def save_model(self, model_path):
        self.Q.save_weights("saved_networks/model.h5", overwrite=True)
    
    def load_model(self):
        self.Q.load_weights("saved_networks/model.h5")
        self.Q.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print ("Weight load successfully")
        
    def choose_action(self, status):
        action = np.zeros(ACTIONS)
        if random.random() <= self.epsilon:
            q = self.Q.predict(status)
            action_index = np.argmax(q)
            action[action_index] = 1
        else:
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        return action, action_index
    
    def process_frame(self, observation, reshape):
        observation = cv2.cvtColor(cv2.resize(observation, (80,80)), cv2.COLOR_BGR2GRAY)
        _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)

        observation = observation / 255.0

        if reshape == True:
            observation = np.reshape(observation, (1, 80, 80, 1))
        return observation
    
    def start_learn(self, buffer):
        train_status, train_action, train_reward, train_next_status, train_terminal = buffer.sample1(BATCH)
        
        # predict value
        next_statue_Q = self.Q_.predict(train_next_status)
        next_statue_Q[range(BATCH), train_action] = train_reward + self.gamma * np.max(next_statue_Q, axis=1) * np.invert(train_terminal)
       
        # train model with mini batch
        self.Q.train_on_batch(train_status, next_statue_Q)
    
    def train_model(self, buffer):
        game = GameState()
        
        for episode in range(100000):
            self.epsilon_step += 1

            # start to play game, first step do nothing
            observation, _, _ = game.play([1, 0])

            # get observation
            observation = self.process_frame(observation, False)

            # get status using four frame
            status = np.stack((observation, observation, observation, observation), axis=2)
            status = np.reshape(status, (1, 80, 80, 4))

            while True:
                # choose action
                action, action_index = self.choose_action(status)

                if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
                    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                # get new observation
                observation_, reward, terminal = game.play(action)

                observation_ = self.process_frame(observation_, True)
                
                # store experience
                next_status = np.append(observation_, status[:,:,:,:3], axis=3)
                buffer.add1((status, action_index, reward, next_status, terminal))

                #if self.time_step % 10 == 0:
                print("train, steps", self.time_step, "/epsilon", self.epsilon, "/action_index", action, "/reward", reward)
                
                self.time_step += 1
                status = next_status
                
                if self.time_step > EXPLORE:
                    self.start_learn(buffer)

                if terminal:
                    game.reset()
                    break
            
            if self.epsilon_step % 1000 == 0:
                self.save_model('saved_networks/flybird')