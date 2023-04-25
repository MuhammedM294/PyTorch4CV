import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

class FMNISTDataset(Dataset):
    def __init__(self,x,y):
        x= x.float()                #convert the images to a float point numbers
        x = x.view(-1,28*28)        #flatten each image
        self.x , self.y = x,y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        x,y = self.x[index], self.y[index]
        return x.to(device), y.to(device) 
    
def build_model(optimizer = SGD, lr = 1e-2 ):
    
    model = nn.Sequential(
                nn.Linear(28 * 28, 1000),
                nn.ReLU(),
                nn.Linear(1000,10)
                         ).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr = lr)   #initialize the learning rate to 0.01
    
    return model , loss_function, optimizer

def train_batch(input_batch, targets, model, loss_function, optimizer):
    model.train()                                        
    predictions = model(input_batch)                     #perfom forward propagation
    batch_loss = loss_function(predictions,targets)      #compute the loss value 
    batch_loss.backward()                                #perfom backward propagation
    optimizer.step()                                     #calculate new weights
    optimizer.zero_grad()                                #flush the gradients for next batch of calculations
    
    return batch_loss.item()                             #return the loss value


@torch.no_grad()                       #disable the gradients calculation
def get_accuracy(input_batch, targets, model):
    model.eval()
    predictions = model(input_batch)              #perfom forward propagation
    max_values, argmaxes = predictions.max(-1)    #get argmax index for each row
    is_correct = argmaxes ==targets               #compare argmax with tragets to check that each row predicts correctly
    
    return is_correct.cpu().numpy().tolist()      #return the result,register it to the cpu, and convert it to numpy array


def train(epoch_number, train_data_loader,model,loss_function, optimizer ):
    losses, accuracies = [], []                      # define lists to contain the loss and accuracy values of each epoch
    for epoch in range (epoch_number):               #define the number of the epochs
        print(f"Epoch: {epoch+1}")
        batch_losses, batch_accuracies = [],[]       #define lists to contain the loss and accuracy values of each batch

        for batch in (iter(train_data_loader)):      #create batches of training data by iterating thorugh data loader
            input_batch, targets = batch                     #unpack the batch 
            batch_loss = train_batch(input_batch, targets, model,loss_function, optimizer)  #train the batch

            batch_losses.append(batch_loss)                               #store the loss value of each batch
        epoch_loss = np.array(batch_losses).mean()                        #get the mean of the loss values of all batches
        losses.append(epoch_loss)
        print(f"Train Loss: {epoch_loss:0.3f}")

        for batch in (iter(train_data_loader)):
            input_batch, targets = batch
            is_correct = get_accuracy(input_batch, targets, model)

            batch_accuracies.extend(is_correct)
        epoch_accuracy = np.mean(batch_accuracies)
        accuracies.append(epoch_accuracy)
        print(f"Train Accuracy: {epoch_accuracy*100:0.0f}%")
        print('<--------------------------------------------------------->')
        
    return losses, accuracies