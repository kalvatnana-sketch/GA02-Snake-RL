"""
store all the agents here
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim






class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=4, use_target_net=True,
                 version=''):
        """ initialize the agent
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values

    Attributes
    ----------
    _model : torch.nn.module
        Q network used to estimate action values
    _target_net : torch.nn.module
        Stores the target Q network, updates periodically
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=4, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_models()

    def reset_models(self): #initalizing q and target network 
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model().to(self.device)
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4) #optimizer
        if(self._use_target_net):
            self._target_net = self._agent_model().to(self.device)
            self._target_net.eval()
            self.update_target_net()

    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : torch.Tensor
            The board state to process

        Returns
        -------
        board : pytorch tensor
            Processed and normalized board
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board)
        if not isinstance(board, torch.Tensor):
            board = torch.as_tensor(board,dtype=torch.float32) #make it into a pytorch tensor
        board = board.permute(0,3,1,2) #adjust channels to pytorch
        board = board.to(self.device)
        return board

    def _get_model_outputs(self, board, model=None): #normalising and reshaping input if needed + forward pass
        """Get Q values predictions from the DQN model for the goven state. 
        """
        # to correct dimensions and normalize
        if not isinstance(board, torch.Tensor):
            board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        return model(board)
   
    def get_qvalues_numpy(self, board):
        out = self._get_model_outputs(board, self._model)
        return out.detach().cpu().numpy()[0].astype(float)



    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Pytorch tensor
            The copy of board state after normalization
        """
        if not isinstance(board, torch.Tensor):
            board = torch.as_tensor(board,dtype=torch.float32)
            
        board = board / 4.0
        board = board.to(self.device)
        return board

    #def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Torch tensor
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
       # model_outputs = self._get_model_outputs(board, self._model)
       # legal_moves = torch.as_tensor(legal_moves, dtype=torch.float32, device=model_outputs.device)
        #return torch.argmax(torch.where(legal_moves==1, model_outputs, -float("inf")), dim=1)

    def move(self, board, legal_moves, value=None): # computing q values and mask out illegal moves
        model_outputs = self._get_model_outputs(board, self._model)
        legal_moves = torch.as_tensor(legal_moves, dtype=torch.float32, device=model_outputs.device)
        masked = torch.where(legal_moves == 1, model_outputs, torch.tensor(-1e9, device=self.device))
        action = torch.argmax(masked, dim=1)
        if action.numel() > 1:
            return action.detach().cpu().numpy()
        return int(action.item())





    def _agent_model(self): #builds CNN from the chosen JSON config 
        """Returns the model which evaluates Q values for a given state input
        Returns
        -------
        model : Pytorch model object
        """
        # define the input layer, shape is dependent on the board size and frames
        import os
        config_path = os.path.join(os.path.dirname(__file__), "model_config", f"{self._version}.json")
        with open(f'model_config/{self._version}.json', 'r') as f:
            m = json.loads(f.read())

        layers = []
        in_channels = self._n_frames

        for layer, params in m['model'].items():
                # add convolutional layer         
            if('Conv2D' in layer):
                filters = params['filters']
                kernel_size = params['kernel_size']
                padding = kernel_size[0]//2 if params.get('padding', '') == 'same' else 0

                layers.append(nn.Conv2d(
                    in_channels= in_channels, 
                    out_channels= filters, 
                    kernel_size = kernel_size,
                    padding= padding

                ))
                if params.get('activation', None)== 'relu':
                    layers.append(nn.ReLU())

                in_channels = filters
                #flatten
            elif('Flatten' in layer):
                layers.append(nn.Flatten())
            
            #dense
            elif 'Dense' in layer:
                units = params['units']
                layers.append(nn.LazyLinear(units))
                if params.get('activation', None) == 'relu':
                    layers.append(nn.ReLU())

        #output
        layers.append(nn.LazyLinear(self._n_actions))
        return nn.Sequential(*layers)
                
    



    def get_action_proba(self, board, values=None):
        out = self._get_model_outputs(board, self._model)
        out = out.detach().cpu().numpy()[0]  # convert to numpy row
        out = np.exp(out - np.max(out))
        return out / np.sum(out)


    def save_model(self, file_path='', iteration=None):
        """Saving model wight with pytorch
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(),
                   "{}/model_{:04d}.pt".format(file_path, iteration))
        if(self._use_target_net):
            torch.save(self._target_net.state_dict(),"{}/model_{:04d}_target.pt".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using pytorchs
        inbuilt load model function (model saved in pt format)
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        path_model = "{}/model_{:04d}.pt".format(file_path, iteration)
        self._model.load_state_dict(torch.load(path_model))
        if(self._use_target_net):
            path_target = "{}/model_{:04d}_target.pt".format(file_path, iteration)
            self._target_net.load_state_dict(torch.load(path_target))

    def print_models(self): 
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model) # changing summary with direct printing of model
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net) #Could have used torchinfo summary for this, but 
            #choose to stay with direct printing to avoid the need to install extra modules when its handed in.

    def train_agent(self, batch_size=64, num_games=1, reward_clip=False): #tried to change batchsize from 32 to 64 to allign with training
        # sample minibatch from replay buffer
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)

        #Converting replay buffer outputs from numpy to tensors
        s_torch = torch.from_numpy(s).float().permute(0, 3, 1, 2).to(self.device) #states
        a_torch_oh = torch.from_numpy(a).float().to(self.device)   # already one-hot
        a_torch = torch.argmax(a_torch_oh, dim=1)                  # extract index
        r_torch = torch.from_numpy(r).float().to(self.device) #rewards
        next_s_torch = torch.from_numpy(next_s).float().permute(0, 3, 1, 2).to(self.device) #next states
        done_torch = torch.from_numpy(done).float().to(self.device) # done flags
        legal_moves_torch = torch.from_numpy(legal_moves).float().to(self.device) #legal moves


        if(reward_clip):
            r_torch = torch.sign(r_torch)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s_torch, current_model)
        # our estimate of expexted future discounted reward
# our estimate of expected future discounted reward
        discounted_reward = r_torch + \
            (self._gamma * torch.max(
                torch.where(
                    legal_moves_torch == 1,
                    next_model_outputs,
                    torch.tensor(-float("inf"), device=self.device)
                ),
                dim=1
            ).values.unsqueeze(1)) * (1 - done_torch)

        # Ensure discounted_reward is (batch, 1)
        if discounted_reward.dim() == 1:
            discounted_reward = discounted_reward.unsqueeze(1)

        # Expand to (batch, n_actions)
        discounted_reward = discounted_reward.expand(-1, self._n_actions)

        # create the target tensor replace only the chosen actions q-value
        with torch.no_grad():
            target = self._get_model_outputs(
        s_torch,
        self._target_net if self._use_target_net else self._model
    )
# modify the selected action values
        target = (1 - a_torch_oh) * target + a_torch_oh * discounted_reward

        # forward pass to get predicted q values
        predicted = self._get_model_outputs(s_torch)
        #loss and update network
        loss_function = torch.nn.MSELoss()
        loss = loss_function(predicted,  target) #pytorch functions instead of train on batch
        # optimizers
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
       # return loss
        return loss.detach()



    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if(self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict()) #insetad of set_weights and get_weights load_state_dict will be used to copy weights in pytorch



