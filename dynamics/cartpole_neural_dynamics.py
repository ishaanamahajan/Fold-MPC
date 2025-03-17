import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import pickle
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
CONFIG = {
    'data': {
        'num_episodes': 10000,  # Number of episodes to collect
        'data_file': 'data/cartpole_data.pkl',  # Add data directory to path
    },
    'model': {
        'hidden_layers': [64, 64, 64],  # Hidden layer sizes
        'learning_rate': 0.001,
        'batch_size': 512,
        'num_workers': 4,
        'epochs': 100,
        'train_val_split': 0.8,
    },
    'save_dir': 'models',
}

def collect_cartpole_data(num_episodes=10000, save_path='data/cartpole_data.pkl'):
    """Collect data from random policy interactions with CartPole"""
    env = gym.make("CartPole-v1", render_mode=None)
    
    data = []
    total_steps = 0
    
    for i in tqdm(range(num_episodes), desc="Collecting episodes"):
        episode = []
        obs, _ = env.reset()
        
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # Store transition: (state, action, next_state)
            transition = np.concatenate([obs, [action], next_obs])
            episode.append(transition)
            
            obs = next_obs
            total_steps += 1
        
        data.append(np.array(episode))
    
    print(f"Collected {total_steps} total steps across {num_episodes} episodes")
    
    # Create data directory if it doesn't exist
    directory = os.path.dirname(save_path)
    if directory:  # Only try to create directory if path is not empty
        os.makedirs(directory, exist_ok=True)
    
    # Save the data
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    return data

class CartpoleDynamicsDataset(Dataset):
    def __init__(self, data, train=True, split=0.8):
        """
        Dataset for training dynamics model
        
        Args:
            data: List of episode trajectories
            train: Whether this is training set (True) or validation set (False)
            split: Training/validation split ratio
        """
        self.state_dim = 4  # Cartpole has 4 state dimensions
        self.action_dim = 1  # Cartpole has 1 action dimension
        
        all_transitions = []
        for episode in data:
            all_transitions.append(episode)
        
        # Concatenate all episodes
        all_data = np.concatenate(all_transitions, axis=0)
        
        # Shuffle the data
        np.random.shuffle(all_data)
        
        # Split into train and validation
        split_idx = int(len(all_data) * split)
        if train:
            self.data = all_data[:split_idx]
        else:
            self.data = all_data[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        transition = self.data[idx]
        
        # Extract state, action, next_state
        state = transition[:4]  # First 4 elements are the state
        action = transition[4:5]  # Next element is the action
        next_state = transition[5:]  # Last 4 elements are the next state
        
        return {
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(action),
            'next_state': torch.FloatTensor(next_state)
        }

class CartpoleDynamicsModel(nn.Module):
    def __init__(self, hidden_layers=[64, 64]):
        """
        Neural dynamics model for Cartpole
        
        Args:
            hidden_layers: List of hidden layer sizes
        """
        super(CartpoleDynamicsModel, self).__init__()
        
        self.state_dim = 4
        self.action_dim = 1
        
        # Build network layers
        layers = []
        input_dim = self.state_dim + self.action_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer predicts the change in state (delta)
        layers.append(nn.Linear(input_dim, self.state_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        """
        Forward pass to predict next state
        
        Args:
            state: [B, state_dim] tensor of states
            action: [B, action_dim] tensor of actions
            
        Returns:
            next_state: [B, state_dim] tensor of predicted next states
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Predict state change (delta)
        delta = self.network(x)
        
        # Add delta to current state to get next state (residual connection)
        next_state = state + delta
        
        return next_state

def train_dynamics_model(data, config):
    """Train the neural dynamics model"""
    # Create datasets
    train_dataset = CartpoleDynamicsDataset(data, train=True, split=config['model']['train_val_split'])
    val_dataset = CartpoleDynamicsDataset(data, train=False, split=config['model']['train_val_split'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['model']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=config['model']['num_workers']
    )
    
    # Create model
    model = CartpoleDynamicsModel(hidden_layers=config['model']['hidden_layers'])
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create directory for saving models
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(config['model']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['model']['epochs']}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for batch in train_pbar:
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            next_state = batch['next_state'].to(device)
            
            # Forward pass
            pred_next_state = model(state, action)
            
            # Compute loss
            loss = criterion(pred_next_state, next_state)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                state = batch['state'].to(device)
                action = batch['action'].to(device)
                next_state = batch['next_state'].to(device)
                
                # Forward pass
                pred_next_state = model(state, action)
                
                # Compute loss
                loss = criterion(pred_next_state, next_state)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_dynamics_model.pt'))
            print(f"New best model saved! (val_loss: {val_loss:.6f})")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config['save_dir'], 'final_dynamics_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Dynamics Model Training')
    plt.legend()
    plt.savefig(os.path.join(config['save_dir'], 'training_curve.png'))
    plt.close()
    
    return model

def test_dynamics_model(model, episodes=5, render=True):
    """Test the dynamics model by predicting next states and comparing to actual"""
    env = gym.make("CartPole-v1", render_mode='human' if render else None)
    model.eval()  # Add this line to put model in evaluation mode
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_error = 0
        num_steps = 0
        
        terminated = truncated = False
        while not (terminated or truncated) and num_steps < 200:  # Updated termination check
            if render:
                env.render()
            
            # Get random action
            action = env.action_space.sample()
            
            # Get model prediction
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor([action]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_next_state = model(state_tensor, action_tensor).cpu().numpy()[0]
            
            # Get actual next state
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # Compute error
            error = np.mean((pred_next_state - next_obs) ** 2)
            total_error += error
            
            # Update state
            obs = next_obs
            num_steps += 1
        
        avg_error = total_error / num_steps
        print(f"Episode {episode+1}, Avg MSE: {avg_error:.6f}, Steps: {num_steps}")
    
    env.close()

def main():
    # Check if data exists, otherwise collect it
    if os.path.exists(CONFIG['data']['data_file']):
        print(f"Loading existing data from {CONFIG['data']['data_file']}")
        with open(CONFIG['data']['data_file'], 'rb') as f:
            data = pickle.load(f)
    else:
        print("Generating new training data...")
        data = collect_cartpole_data(
            num_episodes=CONFIG['data']['num_episodes'],
            save_path=CONFIG['data']['data_file']
        )
    
    # Train dynamics model
    print("Training neural dynamics model...")
    model = train_dynamics_model(data, CONFIG)
    
    # Test the model
    print("Testing dynamics model predictions...")
    test_dynamics_model(model, episodes=5, render=True)

if __name__ == "__main__":
    main()