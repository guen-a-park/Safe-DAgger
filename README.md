# Safe-DAgger
[Video](https://youtu.be/M91-O1PvHL4)
[Paper](https://drive.google.com/file/d/115U6QWEONwgyqQpESF1_1JEwEKVDaTQZ/view?usp=sharing)

This project only has been tested on ubuntu 20.04 environment

If you do want to test other environments' imitation learning behavior check this repository. 


## Requirements

All the requirements except python are written in requirements.txt

python (3.5.6)

tensorflow (1.13.1)

keras (2.3.1)

mujoco (1.50.1.1)

gym (0.17.2)

```sh
If you do not want to use anaconda virtual environment and already install the programs under this block, revise the requirements.txt text file.
```

numpy (1.18.5)

scikit-learn (0.22.2.post1)

cython (0.29.26)

glfw (2.5.0)



**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: If you have problem with the line "you need to install mujoco_py..." 
check https://github.com/openai/mujoco-py/ and see the **Ubuntu installation troubleshooting**


## Getting Started

1. Create a new Conda environment based on Python 3.5. Then, activate it.
```sh
conda create -n (your own env name) python=3.5
conda activate (your own env name)
```
2. Clone this repository

2. Install mujoco-py
    1. Get mujoco license key file from <a href="https://www.roboti.us/license.html">its website</a>
    2. Create a .mujoco folder in the home directory and copy the given mjpro150 directory and your license key into it
      ```sh
      mkdir ~/.mujoco/
      cd <location_of_your_license_key>
      cp mjkey.txt ~/.mujoco/
      cd <this_repo>/mujoco
      cp -r mjpro150 ~/.mujoco/
      ```
    3. Add the following line to bottom of your .bashrc file: 
      ```sh
      export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/
      ```
    
3. Install rest of the libraries given in requirements.txt file using pip
 ```sh
 pip install --user --requirement requirements.txt
 ```

## File explanation

- **expert_bc_cheetah.py** : You can run expert policy, behavior cloning policy and random action.

If you want to run random action, check the bottom of the line and change the function run_exp_bc to run_random.

- **safe_dagger_cheetah.py** : Before you run this file check the folder models2 and find the expert and behavior cloning policy file name as 'HalfCheetah-v1_bc_model.h5' and 'HalfCheetah-v1_expert_model.h5' 

If there are no policy files, run **expert_bc_cheetah.py** first.

During the running, you can see 4 times safe-Dagger iteration and get safe-dagger policy file.


## Function explanation

- Funtion **run_exp_bc** in **expert_bc_cheetah.py** can save expert policy and get expert data from pickle files in the experts folder.

Then, using expert data, it trains behavior cloning policy and save the policies at 'models2' folder.

- Function **run_dagger** in **safe_dagger_cheetah.py** can run safe-Dagger. 

During the first loop, the function **check_diff** checks the differences between 'expert action' and 'behavior cloning action' and if there's any value which the differences are larger than 0.4, print the action as expert action and aggregate those data.

During the second loop and so on, the function **check_diff** check the difference between 'expert aciton' and 'Safe-Dagger action' and check the difference between those actions.



## References

https://github.com/rudolfsteiner/DAgger

https://github.com/berkeleydeeprlcourse/homework_fall2019/tree/master/hw1
