import cv2
import random
import numpy as np

import tensorflow.compat.v1 as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from games import GameState
from models import ReplayMemory

tf.compat.v1.disable_eager_execution()

OBSERVE = 1000.
EXPLORE = 2000000.
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0009
ACTIONS = 2
BATCH = 32

class DeepNetwork:
    def __init__(self, lr=1.0e-6, gamma=0.99, model_path=None):
        self.gamma = gamma
        self.tau = 0.01
        self.time_step = 0
        self.epsilon_step = 0
        self.learning_rate = lr
        self.current_state = None
        self.epsilon = INITIAL_EPSILON

        self.bias_init = tf.constant_initializer(0.01)
        self.kernel_init = tf.random_normal_initializer(mean=0, stddev=0.01)
        
        # state input
        self.obs = tf.placeholder(tf.float32, shape=[None, 80, 80, 4])
        self.obs_= tf.placeholder(tf.float32, shape=[None, 80, 80, 4])

        # action input
        self.action  = tf.placeholder(tf.float32, shape=[None, ACTIONS])
        self.action_ = tf.placeholder(tf.float32, shape=[None, ACTIONS])
        
        # Q value layer
        self.Q = self.create_model(self.obs, scope='eval', trainable=True)
        self.Q_= self.create_model(self.obs_, scope='target', trainable=False)

        # Q value
        self.Q_target = tf.placeholder(tf.float32, [None])

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        # predict Q value
        pred_q_value = tf.reduce_sum(tf.multiply(self.Q, self.action), reduction_indices=1)
        self.q_loss = tf.losses.mean_squared_error(labels=self.Q_target, predictions=pred_q_value)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.q_loss, var_list=self.qe_params)

        # update Q
        self.update_old_Q = [oldq.assign((1 - self.tau) * oldq + self.tau * p) for p, oldq in zip(self.qe_params, self.qt_params)]
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()

        if model_path:
            self.restore_model(model_path)
    
    def create_model(self, input, scope, trainable):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            conv1 = Conv2D(filters=32, kernel_size=[8,8], strides=4, activation='relu', kernel_initializer=self.kernel_init,\
                                            bias_initializer=self.bias_init, trainable=trainable)(input)
            
            pool1 = MaxPooling2D(pool_size=[2,2], strides=2)(conv1)
            
            conv2 = Conv2D(filters=64, kernel_size=[4,4],strides=2, activation='relu', kernel_initializer=self.kernel_init,\
                                            bias_initializer=self.bias_init, trainable=trainable)(pool1)
            
            conv3 = Conv2D(filters=64, kernel_size=[3,3],strides=1, activation='relu', kernel_initializer=self.kernel_init,\
                                            bias_initializer=self.bias_init, trainable=trainable)(conv2)
            
            conv3_flat = Flatten()(conv3)
            
            fc = Dense(units=512, activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init, \
                                            trainable=trainable)(conv3_flat)
            
            q = Dense(units=2, kernel_initializer=self.kernel_init, bias_initializer=self.bias_init, trainable=trainable)(fc)
            return q
    
    def save_model(self, model_path, global_step):
        self.saver.save(self.session, model_path, global_step=global_step)
    
    def restore_model(self, model_path):
        checkpoint = tf.train.get_checkpoint_state(model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            self.epsilon_step = int(checkpoint.model_checkpoint_path.split('-')[-1])
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        
    def choose_action(self, status):
        action = np.zeros(ACTIONS)
        if random.random() <= 1 - self.epsilon:
            action_index = np.argmax(self.session.run(self.Q, {self.obs:[status]})[0])
            action[action_index] = 1
        else:
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        return action
    
    def process_frame(self, observation, reshape):
        observation = cv2.cvtColor(cv2.resize(observation, (80,80)), cv2.COLOR_BGR2GRAY)
        _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)

        observation = observation / 255.0

        if reshape == True:
            observation = np.reshape(observation, (80, 80, 1))
        return observation
    
    def start_learn(self, buffer):
        train_status, train_action, train_reward, train_next_status, train_terminal = buffer.sample(BATCH)
        next_target_Q = self.session.run(self.Q_, {self.obs_:train_next_status})

        target_q=[]
        for i in range(len(train_terminal)):
            if train_terminal[i]:
                target_q.append(train_reward[i])
            else:
                target_q.append(train_reward[i] + self.gamma * np.max(next_target_Q[i]))
        
        self.session.run(self.train_step, feed_dict={self.obs:train_status, self.action:train_action, self.Q_target:target_q})
        
        self.session.run(self.update_old_Q)
    
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

            while True:
                # choose action
                action = self.choose_action(status)

                if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
                    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                # get new observation
                observation_, reward, terminal = game.play(action)
                observation_ = self.process_frame(observation_, True)
                
                # store experience
                next_status = np.append(observation_, status[:,:,:3], axis=2)
                #experience = np.reshape(np.array([status, action, reward, next_status, terminal]), [1, 5])
                buffer.add([status, action, reward, next_status, terminal])
                
                if self.time_step > OBSERVE:
                    self.start_learn(buffer)
                
                # update current status with new
                self.time_step += 1
                status = next_status

                # print info
                if self.time_step % 1000 == 0:
                    print("train, steps", self.time_step, "/epsilon", self.epsilon, "/action_index", action, "/reward", reward)

                # reset if terminal
                if terminal:
                    game.__init__()
                    break
            
            # save model -- change 100 to 100000 or other number. 
            print('epsilon step --> ', self.epsilon_step)
            if self.epsilon_step % 10 == 0:
                self.save_model('saved_networks/flybird', global_step = self.epsilon_step)