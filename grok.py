import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import deque
import random
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)   # After pooling: 32x32 -> 8x8 -> 4x4
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))

        x = self.fc2(x)
        return x


# Edge Device Class
class EdgeDevice:
    def __init__(self, id, cpu_freq, energy, bandwidth):
        self.id = id
        self.cpu_freq = cpu_freq  # Hz
        self.energy = energy      # Energy units
        self.bandwidth = bandwidth  # Mbps
        self.model = CNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.local_data = None

    def set_local_data(self, data):
        self.local_data = data

    def train_local_model(self, epochs):
        if self.local_data is None or len(self.local_data) < 32:
            print(f"Device {self.id}: Insufficient data ({len(self.local_data) if self.local_data else 0} samples), skipping training.")
            return self.model.state_dict(), 0, 0, 0

        if len(self.local_data) < 32:
            print(f"Device {self.id}: Local data size {len(self.local_data)} too small for batch_size=32. Skipping training.")
            return self.model.state_dict(), 0, 0, 0

        self.model.train()
        loader = torch.utils.data.DataLoader(self.local_data, batch_size=32, shuffle=True, drop_last=True)
        
        for epoch in range(epochs):
            for data, target in loader:
                # Debug print to verify batch sizes
                print(f"Device {self.id}, Epoch {epoch}: Input batch size = {data.size(0)}, Target batch size = {target.size(0)}")
                self.optimizer.zero_grad()
                output = self.model(data)
                print(f"Device {self.id}, Epoch {epoch}: Output shape = {output.shape}, Target shape = {target.shape}")

                loss = self.criterion(output, target.view(-1))
                loss.backward()
                self.optimizer.step()

        # Calculate computational time and energy consumption
        G = 1e6  # CPU cycles per bit
        mu = len(self.local_data)
        T_local = mu * G / self.cpu_freq
        tau = 1e-9  # Switched capacitance
        B_k = (self.cpu_freq ** 2) * tau * mu * G
        self.energy -= B_k
        
        # Transmission time
        D = 1e6  # Model size in bits
        T_trans = D / self.bandwidth
        
        return self.model.state_dict(), T_local, T_trans, B_k

    def report_resources(self):
        return {'cpu_freq': self.cpu_freq, 'energy': self.energy, 'bandwidth': self.bandwidth}

# MEC Server Class
class MECServer:
    def __init__(self, num_devices=10):
        self.global_model = CNN()
        self.devices = [EdgeDevice(i, cpu_freq=np.random.uniform(1e9, 3e9),
                                  energy=np.random.uniform(100, 500),
                                  bandwidth=np.random.uniform(10, 50)) 
                        for i in range(num_devices)]
        self.energy_history = []
        self.bandwidth_history = []

    def distribute_data(self):
        data_per_device = len(trainset) // len(self.devices)
        print(f"Distributing {data_per_device} samples per device")
        for i, device in enumerate(self.devices):
            subset = torch.utils.data.Subset(trainset, range(i * data_per_device, (i + 1) * data_per_device))
            device.set_local_data(subset)

    def fed_avg(self, local_weights, data_sizes):
        total_size = sum(data_sizes)
        avg_weights = {key: torch.zeros_like(val) for key, val in local_weights[0].items()}
        for weights, size in zip(local_weights, data_sizes):
            for key in weights:
                avg_weights[key] += weights[key] * (size / total_size)
        return avg_weights

    def simulate_training_round(self, selected_devices, epochs=1):
        local_weights = []
        data_sizes = []
        total_energy = 0
        total_bandwidth = 0
        
        for device_id in selected_devices:
            device = self.devices[device_id]
            weights, T_local, T_trans, energy_used = device.train_local_model(epochs)
            local_weights.append(weights)
            data_sizes.append(len(device.local_data))
            total_energy += energy_used
            total_bandwidth += device.bandwidth
        
        # Aggregate models
        new_weights = self.fed_avg(local_weights, data_sizes)
        self.global_model.load_state_dict(new_weights)
        
        # Store metrics
        self.energy_history.append(total_energy)
        self.bandwidth_history.append(total_bandwidth)
        
        return T_local + T_trans

    def evaluate_global_model(self):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        return accuracy

# DDQN Class
class DDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Device Selection Function
def select_devices_ddqn(mec_server, ddqn, num_devices_to_select):
    state = np.array([list(d.report_resources().values()) for d in mec_server.devices]).flatten()
    state = state.reshape(1, -1)
    selected = []
    for _ in range(num_devices_to_select):
        action = ddqn.act(state)
        if action not in selected:
            selected.append(action)
        next_state = state  # Simplified
        reward = 1  # Simplified
        ddqn.remember(state, action, reward, next_state, False)
        state = next_state
    return selected

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Main Simulation
if __name__ == "__main__":
    # Initialize MEC server and DDQN
    mec_server = MECServer(num_devices=10)
    mec_server.distribute_data()
    ddqn = DDQN(state_size=30, action_size=10)  # 3 features * 10 devices

    # Training loop
    num_rounds = 20
    for round_num in range(num_rounds):
        selected_devices = select_devices_ddqn(mec_server, ddqn, num_devices_to_select=5)
        delay = mec_server.simulate_training_round(selected_devices, epochs=1)
        accuracy = mec_server.evaluate_global_model()
        print(f"Round {round_num+1}: Accuracy = {accuracy:.4f}, Delay = {delay:.4f}")
        
        if len(ddqn.memory) > 32:
            ddqn.replay(32)
        ddqn.update_target_model()

    # Plot energy and bandwidth graphs
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(mec_server.energy_history, label='Energy Consumption')
    plt.xlabel('Training Round')
    plt.ylabel('Energy Units')
    plt.title('Energy Consumption Over Rounds')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mec_server.bandwidth_history, label='Bandwidth Usage')
    plt.xlabel('Training Round')
    plt.ylabel('Bandwidth (Mbps)')
    plt.title('Bandwidth Usage Over Rounds')
    plt.legend()

    plt.tight_layout()
    plt.show()
