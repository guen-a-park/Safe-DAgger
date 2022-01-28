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


expert_name = "HalfCheetah-v1"
data_file = "data/HalfCheetah-v2_20_data.pkl"

#calculate the difference
def check_diff(exp_act,bc_act): 
    x = np.array([exp_act,bc_act])
    diff = np.squeeze(np.diff(x, axis=0))
    #하나라도 일정값 이상 차이나면 exp_action으로 저장
    for i in diff:
        if i > 0.001:
            return i
        else: return 0

#run dagger
def run_dagger(expert_name, expert_data_file, render = True):

    print('dagger #0')
    #dagger first loop
    with tf.Session():
        tf_util.initialize()

        env = gym.make("HalfCheetah-v2") # "Hopper-v1"->'Hopper-v2' version issue
        max_steps = env.spec.max_episode_steps #change timestep_limit to max_episode_steps (version issue)

        returns = []
        dagger_observations = []
        dagger_actions = []
        
        #env.seed(1234) 
        obs = env.reset()

        done = False
        totalr = 0.
        steps = 0
        j=0
        k=0

        expert_model = load_model('models2/' + expert_name + '_expert_model.h5')
        cloned_model = load_model('models2/' + expert_name + '_bc_model.h5')

        while not done:

            exp_action = expert_model.predict(obs[None, :], batch_size = 64, verbose = 0)
            bc_action = cloned_model.predict(obs[None, :], batch_size = 64, verbose = 0)

            #check difference
            tau = check_diff(exp_action,bc_action)

            if tau>0.001:
                dagger_observations.append(obs)
                dagger_actions.append(exp_action)
                obs, r, done, _ = env.step(exp_action)
                j+=1
            else :
                obs, r, done, _ = env.step(bc_action)
                k+=1

            totalr += r
            steps += 1

            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

        print('exp_action:', j,'bc_action :', k)
        returns.append(totalr)

        print(np.shape(dagger_actions))

        print('returns', returns)
        #print('mean return', np.mean(returns))
        #print('std of return', np.std(returns))


    #load expert data
    load_npz = np.load('exp_data/' + expert_name + ' act_obs.npz')
    #print(load_npz.files)
    obs_data = load_npz['x']
    act_data = load_npz['y']

    load_npz.close()

    #data aggregation
    dagger_actions = np.array(dagger_actions)
    obs_data = np.concatenate((obs_data, np.array(dagger_observations)))
    act_data = np.concatenate((act_data, np.array(dagger_actions.reshape(dagger_actions.shape[0], dagger_actions.shape[2]))))
    
    print(np.shape(act_data))
    print(np.shape(obs_data))

    for p in range(3): #dagger loop

        print('dagger #',p+1)
        #create a Feedforward network useing Keras
        #lr=0.001, https://keras.io/api/optimizers/adam/

        model = Sequential()
        model.add(Dense(96, activation = "relu", input_shape = (obs_data.shape[1],)))
        model.add(Dense(96, activation = "relu"))
        model.add(Dense(96, activation = "relu"))
        model.add(Dense(act_data.shape[1], activation = "linear"))
        model.compile(loss = "mean_squared_error", optimizer = "adam", metrics=["accuracy"]) 
        model.fit(obs_data, act_data, batch_size = 64, epochs = 30, verbose = 1)

        model.save('models2/' + expert_name + '_safedagger_model.h5') # safe dagger policy

        with tf.Session():
            tf_util.initialize()

            env = gym.make("HalfCheetah-v2") # "Hopper-v1"->'Hopper-v2' version issue
            max_steps = env.spec.max_episode_steps #change timestep_limit to max_episode_steps (version issue)

            returns = []
            dagger_observations = []
            dagger_actions = []
            
            #env.seed(1234) 
            obs = env.reset()

            #print(obs[None,:])
            #print(env.step(0))
            done = False
            totalr = 0.
            steps = 0
            j=0
            k=0

            expert_model = load_model('models2/' + expert_name + '_expert_model.h5')
            dagger_model = load_model('models2/' + expert_name + '_safedagger_model.h5')

            while not done:

                exp_action = expert_model.predict(obs[None, :], batch_size = 64, verbose = 0)
                dagger_action = dagger_model.predict(obs[None, :], batch_size = 64, verbose = 0)

                #check difference
                tau = check_diff(exp_action,dagger_action)

                if tau>0.001:
                    dagger_observations.append(obs)
                    dagger_actions.append(exp_action)
                    obs, r, done, _ = env.step(exp_action)
                    j+=1
                else :
                    obs, r, done, _ = env.step(dagger_action)
                    k+=1

                totalr += r
                steps += 1

                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break

            print('exp_action:', j,'dagger_action :', k)
            returns.append(totalr)

            print(np.shape(dagger_actions))

            print('returns', returns)
            #print('mean return', np.mean(returns))
            #print('std of return', np.std(returns))

        #data aggregation
        dagger_actions = np.array(dagger_actions)
        
        obs_data = np.concatenate((obs_data, np.array(dagger_observations)))
        act_data = np.concatenate((act_data, np.array(dagger_actions.reshape(dagger_actions.shape[0], dagger_actions.shape[2]))))
        print(np.shape(act_data))
        print(np.shape(obs_data))

run_dagger(expert_name, data_file)



#################################################################################

#check_space

# env = gym.make('HalfCheetah-v2')
# print(env.observation_space) #dim 17
# # print(env.action_space) #dim 6, float
# print(env.observation_space.high)
# print(env.observation_space.low)
# # print(env.action_space.high) # [1. 1. 1. 1. 1. 1.]
# # print(env.action_space.low) # [-1. -1. -1. -1. -1. -1.]


################################################################################

