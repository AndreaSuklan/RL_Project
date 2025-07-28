<div align="center">
    <h1>Analysis of RL Algorithms for a Simulated Hill Climb Racing Agent</h1>
    <h3>Authors: Giovanni Billo, Andrea Suklan and Carlos Velázquez Fernández</h3>
    <h6>Final project of the Reinforcement Learning course - UniTs</h6>
</div>

<div align="center">
    <img src="presentation/images/demo.gif" alt="Video
  Demo" />
</div>

This project is a custom implementation of the classic Hill Climb Racing game using Python, Pygame, and the Box2D physics engine. The primary goal is to train and compare different reinforcement learning agents to master the challenge of navigating an infinitely generated, rugged terrain.

***

## Features

* **Custom Environment:** A fully custom Hill Climb environment built from scratch using `gymnasium` and the `Box2D` physics engine.
* **Multiple RL Algorithms:** Implementation and comparison of three distinct reinforcement learning algorithms:
    * **PPO (Proximal Policy Optimization)**
    * **DQN (Deep Q-Network)**
    * **Expected SARSA**
* **Function Approximation:** Support for different models, including Neural Networks (`nn`) and Polynomial (`poly`) function approximators.
* **Train & Visualize:** A command-line interface to easily train new agents and visualize the performance of saved models.

***

## Algorithms Implemented

This project explores both value-based and policy-based reinforcement learning methods:

* **PPO (Proximal Policy Optimization):** An advanced actor-critic method known for its stability and sample efficiency. It uses a clipped objective function to constrain policy updates.
* **DQN (Deep Q-Network):** A classic value-based algorithm that utilizes an experience replay buffer and a target network to stabilize learning a Q-value function.
* **Expected SARSA:** An on-policy temporal-difference algorithm that improves upon SARSA by calculating the expected Q-value over all possible next actions, reducing variance.

***


## Usage

The project is controlled via the `main.py` script. You can either `train` a new agent or `visualize` a pre-trained one.


### Command-Line Arguments

| Argument | Shorthand | Description | Default Value |
| :--- | :--- | :--- | :--- |
| `action` | | Action to perform (`train` or `visualize`). | **Required** |
| `algorithm`| | Algorithm to use for `train` (`ppo`, `dqn`, `sarsa`). | **Required for train** |
| `--path` | | Path to the model `.zip` file for `visualize`. | **Required for visualize**|
| `--model` | | Model type for the agent (`nn`, `linear`, `poly`). | `nn` |
| `--degree` | | Degree for the polynomial model. | `3` |
| `--buffer_size`| | Buffer size for the PPO algorithm. | `2048` |
| `--episodes` | `-e` | Total episodes for training. | `100` |
| `--seed` | `-s` | Random seed for the run. | `0` |
| `--verbose`| `-v` | Verbosity level (0 or 1). | `0` |


### Training Agents

Use the `train` action followed by the algorithm name.

**Train a PPO agent (recommended):**
```bash
python src/main.py train ppo --buffer_size 4096 --seed 42
```

**Train a DQN agent:**
```
python src/main.py train dqn --model nn --seed 123
```

**Train an Expected SARSA agent with a polynomial model**
```
python src/main.py train sarsa --model poly --degree 2 --seed 10
```


### Visualizing Agents
Use the `visualize` action to see your trained agents in action. The script will automatically find the corresponding saved model in the `/models` directory.

**Visualize the PPO agent:**
```bash
python src/main.py visualize --path models/ppo_nn_2048_42.zip
```

**Visualize the DQN agent:**
```
python src/main.py visualize --path models/dqn_nn_123.zip
```

**Visualize the Expected SARSA agent with a polynomial model**
```
python src/main.py visualize --path models/sarsa_poly_2_10.zip
```

### Plotting the performance
Use the plotting_results script to save the plots of rewards and mean loss per episode, or also the entropy specifically for ppo

**Plot rewards and losses of all the algorithms compared:**
```bash
python .\src\plot_results.py
```

**Select which algorithm to plot with different function approximation methods:**
```bash
python .\src\plot_results.py --alg dqn
```


## Outputs

- **Models**: Trained agent models are saved as `.zip` files in the `/models` directory.

- **Logs**: Training data, such as rewards and episode lengths, are saved as `.csv` files in the `/logs` directory, allowing for performance analysis and plotting.

## Replicating the results
The repo comes with a handy script to replicate all the experiments that were done on orfeo.
The script submits all experiments as a series of SLURM job arrays with different models and seeds, collecting all relevant data for plotting in the logs/ directory, and all the trained models with the different seeds in the models/ directory. All the hyperparameters and training modes are specified in the `rl_params.csv` file, and parsed in the `run.sh` file. This setup is easily extendable to other hyperparameters for further tuning. After adding the, as CLI arguments to `src/main.py`, one just needs to add the specific column to `rl_grid.csv` and a parse line + variable in run.sh.

To run the script, just execute the following command:
```
chmod +x ./submit.sh #in case the file is not executable yet
./submit.sh
``` 
In case your `AssocMaxSubmitJobLimit` in Orfeo or other clusters is too low, the script automatically loops until it can submit all the jobs in the queue, retrying every 3 minutes.
In order to have this process running independently from the state of your machine, it is recommended to use `tmux`.
```
tmux
chmod +x ./submit.sh #in case the file is not executable yet
./submit.sh
``` 
Then hit `Ctrl-B D` to detach the session. 
