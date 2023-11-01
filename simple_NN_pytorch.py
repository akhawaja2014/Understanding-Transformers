#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:23:44 2023

@author: arsalankhawaja
"""

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from tqdm import tqdm

class NN(torch.nn.Module):
    
    def __init__(self,input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = torch.nn.Linear(input_size,1012)
        self.fc2 = torch.nn.Linear(1012,512)
        self.fc3 = torch.nn.Linear(512,num_classes)
        
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


device = torch.device('cpu')

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

    
train_dataset = datasets.MNIST(root = '/Users/arsalankhawaja/Documents/Projects/LearnPytorch',train =True, transform = transforms.ToTensor())
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = '/Users/arsalankhawaja/Documents/Projects/LearnPytorch',train =False, transform = transforms.ToTensor())
test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

model = NN(input_size = input_size, num_classes = num_classes).to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_dataloader)):
        data = data.to(device = device)
        targets = targets.to(device = device)
        #print(data.shape)
        #Now we need to flatten the images
        data = data.reshape(data.shape[0],-1)
        #print(data.shape)
        scores = model(data)
        loss = criterion(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0],-1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
    model.train()
    return num_correct / num_samples
        

# Check accuracy on training & test to see how good our model
print(f"Accuracy on training set: {check_accuracy(train_dataloader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_dataloader, model)*100:.2f}")
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



