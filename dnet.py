#!/usr/bin/env python
# coding: utf-8

# # Dnet
# 
# Assess damaged buildings

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from dnet_dataloader import DamageNetDataset


# ### Config

# In[2]:


BATCH_SIZE = 16
EPOCHS = 57 * 15  # 57 * 10


# ### Data loading

# In[3]:


dataset = DamageNetDataset(images_dir='train/images', labels_dir='train/labels', transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((75, 75)), 
    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.ToTensor()]))


# In[4]:


dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# ### Data Visualization

# In[5]:


def show_img(img, transpose=True):
    # img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    if transpose:
        npimg = npimg.transpose(1, 2, 0)
    plt.imshow(npimg)
    plt.show()


# In[6]:


dataiter = iter(dataloader)
images, labels = next(dataiter)

show_img(images[0])
print(images.shape)

print(labels)
print(labels.shape)


# ### Net

# In[7]:


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(8, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        
        nn.Flatten(),
        nn.Linear(1568, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
    )
  
  def forward(self, x):
        
    """for layer in self.model:
        x = layer(x)
        print(x.size())"""
        
    return self.model(x)

# net = Net()


# ### Training

# In[8]:


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device "%s" for training' % dev)

print(torch.cuda.device_count())

net = Net().to(dev)


# In[9]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# In[10]:


loss_list = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(dev)
        labels = labels.to(dev)
        
        optimizer.zero_grad()
        
        y = net(images)
        
        loss = criterion(y, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        print('[%.3d / %.3d] Loss: %.9f' % (epoch, i, loss.item()))
    
    average_loss = running_loss / 175
    loss_list.append(average_loss)
    
    torch.save(net.state_dict(), 'model/state_dict_model.pt')


# In[11]:


fig = plt.figure()
plt.plot(loss_list)
plt.title('Loss Value Plot')
plt.xlabel('Batches in Epochs')
plt.ylabel('Loss Value')
plt.savefig('model/Loss_Graph.png', bbox_inches='tight', dpi=600)
plt.show()


# ### Evaluate
# 
# - https://www.ibm.com/cloud/blog/the-xview2-ai-challenge
# - [F1 Score with sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
# - [Confusion Matrix with sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

# In[12]:


net = Net()
net.load_state_dict(torch.load('model/state_dict_model.pt'))
net.eval()
net = net.to(dev)


# In[13]:


dataset = DamageNetDataset(images_dir='test/images', labels_dir='test/labels', transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((75, 75)),
    torchvision.transforms.ToTensor()]))


# In[14]:


dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# In[17]:


from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sn
import pandas as pd

predictions = []
class_labels = []


for i, (images, labels) in enumerate(dataloader):
    images = images.to(dev)
    labels = labels.to(dev)

    y = net(images)
    y = torch.sigmoid(y)
    
    loss = criterion(y, labels)
    # print('Loss: ', loss)
    
    # Convert ordinal encoding into class prediction output
    # 0 -> no-damage
    # 1 -> minor-damage
    # 2 -> major-damage
    # 3 -> destroyed
    # y[0] = torch.tensor([1, 1, 0])
    # print(y[0])
    threshhold = 0.5
    
    for i in range(len(y)):
        for j in range(len(y[0])):
            if y[i][j] < threshhold:
                predictions.append(j)
                break
            if j == len(y[0]) - 1:
                predictions.append(3)
    # print(predictions)
    # print(len(predictions))
                
    # Do the same for the labels
    no_damage = torch.tensor([0, 0, 0]).to(dev)
    minor_damage = torch.tensor([1, 0, 0]).to(dev)
    major_damage = torch.tensor([1, 1, 0]).to(dev)
    destroyed = torch.tensor([1, 1, 1]).to(dev)
    
    for i in range(len(labels)):
        if torch.all(torch.eq(labels[i], no_damage)):
            class_labels.append(0)
        elif torch.all(torch.eq(labels[i], minor_damage)):
            class_labels.append(1)
        elif torch.all(torch.eq(labels[i], major_damage)):
            class_labels.append(2)
        elif torch.all(torch.eq(labels[i], destroyed)):
            class_labels.append(3)

    
# Work out accuracy    
# Probably not the best because of imbalanced data
num_correct = 0

for i in range(len(predictions)):
    if predictions[i] == class_labels[i]:
        num_correct = num_correct + 1

accuracy = (num_correct / len(predictions)) * 100
print('Accuracy: ', accuracy, '%')       
    
# Print Confusion Matrix
confusion_matrix = confusion_matrix(class_labels, predictions)
print('Confusion Matrix: ', confusion_matrix)

df_cm = pd.DataFrame(confusion_matrix, range(4), range(4))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap=sn.cm.rocket_r, fmt='d') # font size
plt.savefig('model/Confusion_Matrix.png')
plt.show()
    
# F1 score is the harmonic mean between precision and recall
# Value [0, 1]
f1_score = f1_score(class_labels, predictions, average='weighted')
print('F1-Score: ', f1_score)  

print(predictions)
print(len(predictions))

print(class_labels)
print(len(class_labels))
    


# In[ ]:




