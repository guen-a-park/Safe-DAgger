import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import math

from keras.models import Sequential, load_model
from keras.layers import Dense #, Dropout, Activation, Flatten, Reshape
#from keras.utils import np_utils
from sklearn.utils import shuffle

#run expert data
#python python run_expert.py experts/Humanoid-v1.pkl Humanoid-v2 --render --num_rollouts 20

#use cpu

def load_task_data(filename):
    with open(filename, 'rb') as f:
        task_data = pickle.loads(f.read())
    return task_data

#print(gym.__version__)
expert_name = "HalfCheetah-v1"
data_file = "data/HalfCheetah-v2_20_data.pkl"


#run expert,bc
def run_exp_bc(expert_name, expert_data_file, render = False):
    #expert_name: the gym expert policy name
    #render: True to render

    print('loading and building expert policy')
    expert_policy_file = "./experts/" + expert_name + ".pkl"
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    task_data = load_task_data(expert_data_file)  #"data/" + data_file + ".pkl")
    obs_data = np.array(task_data["observations"])
    act_data = np.array(task_data["actions"])
    act_data = act_data.reshape(act_data.shape[0], act_data.shape[2])

    model = Sequential()
    model.add(Dense(96, activation = "relu", input_shape = (obs_data.shape[1],)))
    model.add(Dense(96, activation = "relu"))
    model.add(Dense(96, activation = "relu"))
    model.add(Dense(act_data.shape[1], activation = "linear"))
    model.compile(loss = "mean_squared_error", optimizer = "adam", metrics=["accuracy"]) 
    model.fit(obs_data, act_data, batch_size = 64, epochs = 30, verbose = 1)

    model.save('models2/' + expert_name + '_expert_model.h5') #save expert policy

    with tf.Session():
        tf_util.initialize()

        env = gym.make("HalfCheetah-v2") # "Hopper-v1"->'Hopper-v2' version issue
        max_steps = env.spec.max_episode_steps #change timestep_limit to max_episode_steps (version issue)

        returns = []
        exp_observations = []
        exp_actions = []

        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0

        expert_model = load_model('models/' + expert_name + '_expert_model.h5')
        while not done:

            exp_action = expert_model.predict(obs[None, :], batch_size = 64, verbose = 0)
            obs, r, done, _ = env.step(exp_action)

            exp_observations.append(obs)
            exp_actions.append(exp_action)

            totalr += r
            steps += 1

            if render: #render expert data
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

        returns.append(totalr)

        print('returns', returns)

    exp_actions = np.array(exp_actions)
    obs_data = np.array(exp_observations)
    act_data = np.array(exp_actions.reshape(exp_actions.shape[0], exp_actions.shape[2]))
    print(np.shape(act_data))
    print(np.shape(obs_data))

    #save expert_data
    np.savez('exp_data/' + expert_name + ' act_obs',x=obs_data,y=act_data)

    #saving bc policy
    model = Sequential()
    model.add(Dense(96, activation = "relu", input_shape = (obs_data.shape[1],)))
    model.add(Dense(96, activation = "relu"))
    model.add(Dense(96, activation = "relu"))
    model.add(Dense(act_data.shape[1], activation = "linear"))
    model.compile(loss = "mean_squared_error", optimizer = "adam", metrics=["accuracy"]) 
    model.fit(obs_data, act_data, batch_size = 64, epochs = 30, verbose = 1)

    model.save('models2/' + expert_name + '_bc_model.h5') # bc policy

    with tf.Session():
        tf_util.initialize()

        env = gym.make("HalfCheetah-v2") # "Hopper-v1"->'Hopper-v2' version issue
        max_steps = env.spec.max_episode_steps #change timestep_limit to max_episode_steps (version issue)

        returns = []
        bc_observations = []
        bc_actions = []

        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        cloned_model = load_model('models/' + expert_name + '_bc_model.h5')
        while not done:

            bc_action = cloned_model.predict(obs[None, :], batch_size = 64, verbose = 0)
            obs, r, done, _ = env.step(bc_action)

            bc_observations.append(obs)
            bc_actions.append(bc_action)

            totalr += r
            steps += 1

            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

        returns.append(totalr)

        print('returns', returns)

    print(np.shape(bc_actions))
    print(np.shape(bc_observations))
    # bc_actions = np.array(bc_actions)
    # obs_data = np.array(bc_observations)
    # act_data = np.array(bc_actions.reshape(bc_actions.shape[0], bc_actions.shape[2]))

def run_random(render = True, num_rollouts = 10):
    env = gym.make('HalfCheetah-v2') #"Hopper-v1"->'Hopper-v2' version issue
    max_steps = env.spec.max_episode_steps #change timestep_limit to max_episode_steps (version issue)

    returns = []
    rand_observations = []
    rand_actions = [] #위치

    for i in range(num_rollouts):

        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
    
        while not done: #while or if done == True
            action = np.random.uniform(-1,1,6)
            rand_observations.append(obs)
            rand_actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1

            print(steps)
            if render:
                env.render()
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    return returns

run_exp_bc(expert_name, data_file)
#run_random()



