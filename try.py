import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. إعداد بيانات CIFAR-10 (CNN Model)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. بيئة 5G و MEC و أجهزة IoT
class MDP:
    def __init__(self, num_devices):
        self.num_devices = num_devices
        self.states = [{'cpu': random.uniform(0.5, 2.0), 'energy': random.uniform(1, 5), 'bandwidth': random.uniform(10, 100)} for _ in range(num_devices)]
    
    def get_state(self):
        return np.array([[device['cpu'], device['energy'], device['bandwidth']] for device in self.states])

# تحسين تمثيل شبكة 5G 
class Network5G:
    def __init__(self, num_devices):
        self.num_devices = num_devices
        self.bandwidths = [random.uniform(10, 100) for _ in range(num_devices)]
        self.latencies = [random.uniform(1, 10) for _ in range(num_devices)]
    
    def get_network_state(self):
        return np.array([[self.bandwidths[i], self.latencies[i]] for i in range(self.num_devices)])

# 3. خوارزمية DDQN
class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. إدارة الموارد والطاقة
ALPHA = 0.1  # Energy coefficient
TIME = 1

def compute_energy(cpu):
    return ALPHA * (cpu ** 2) * TIME

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# 5. التدريب والتقييم
num_devices = 5
state_size = 3  # CPU, Energy, Bandwidth
action_size = 2  # Participate (1) or Not (0)
epsilon = 0.1
learning_rate = 0.001
batch_size = 32

mdp = MDP(num_devices)
network_5g = Network5G(num_devices)
q_network = DDQN(state_size, action_size)
target_network = DDQN(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())
buffer = ReplayBuffer()
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

for episode in range(100):
    state = mdp.get_state()
    network_state = network_5g.get_network_state()
    action = [random.choice([0, 1]) for _ in range(num_devices)]
    next_state = mdp.get_state()
    reward = sum(action)
    total_energy = sum(compute_energy(state[i][0]) for i in range(num_devices) if action[i] == 1)
    buffer.push(state, action, reward, next_state)

    loss = None
    if len(buffer.buffer) > batch_size:
        batch = buffer.sample(batch_size)
        for state, action, reward, next_state in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.long)

            q_values = q_network(state_tensor)
            target_q_values = target_network(next_state_tensor).detach()
            expected_q = reward + 0.99 * torch.max(target_q_values, dim=1)[0]

            loss = loss_fn(q_values.gather(1, action_tensor.unsqueeze(1)), expected_q.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())
        if loss is not None:
            print(f"Episode {episode}: Loss={loss.item():.4f}, Energy Consumption={total_energy:.4f} J")
        else:
            print(f"Episode {episode}: No training yet, Energy Consumption={total_energy:.4f} J")

print("Training completed!")
