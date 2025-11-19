# Snake Reinforcement Learning

This is a PyTorch version of the DeepQLearningAgent class for the Snake environment that was originally written in TensorFlow.  
Only the DQN agent is rewritten, while other classes not necessary for this project have been removed.

Below are visualizations generated from my PyTorch-trained DQN model:

<video width="400" controls>
  <source src="images/game_visual_v17.1_163500_9.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video width="400" controls>
  <source src="images/game_visual_v17.1_163500_oos_9.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## Code Structure

### `game_environment.py`
Contains the necessary code to create and interact with the Snake environment (classes `Snake` and `SnakeNumpy`), similar to the OpenAI Gym interface.


### `agent.py`
Contains the PyTorch implementation of the Deep Q-Learning agent.

| Class               | Description                                 |
|--------------------|---------------------------------------------|
| DeepQLearningAgent | Deep Q-Learning algorithm with CNN network  |

### `training.py`
Contains the full training loop.

### `game_visualization.py`
Generates `.mp4` visualizations of gameplay.

---

## Experiments

This project focuses on the PyTorch implementation of Deep Q-learning.  
The model architecture corresponds to version "v17.1" from the original TensorFlow implementation.  
The Adam optimizer is used.  
The resulting gameplay behavior is shown in the visualizations above.

---

# Running Graded Assignment 02

To run the project in the provided GA01 Conda environment:

# 1. Activate the environment
conda activate uitnn

# 2. Navigate to folder
cd GA02-Snake-RL

# 3. Train model
python training.py

# 4.Test the Snake environment
python snake_test_script.py

# 5. Generate visualizations from the trained model
python game_visualization.py


