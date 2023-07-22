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
        #convert the images to a float point numbers
        #divide the input data by the maximum value
        x= x.float()/255 
        #flatten each image
        x = x.view(-1,28*28)        
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
    #perfom forward propagation
    predictions = model(input_batch)
    #compute the loss value 
    batch_loss = loss_function(predictions,targets)
    #perfom backward propagation
    batch_loss.backward()
    #calculate new weights
    optimizer.step()
    #flush the gradients for next batch of calculations
    optimizer.zero_grad()                                
     #return the loss value
    return batch_loss.item()                            


#disable the gradients calculation
@torch.no_grad()                       
def get_accuracy(input_batch, targets, model):
    
    model.eval()
    #perfom forward propagation
    predictions = model(input_batch)
    #get argmax index for each row
    max_values, argmaxes = predictions.max(-1)
    #compare argmax with tragets to check that each row predicts correctly
    is_correct = argmaxes ==targets               
    
    #return the result,register it to the cpu, and convert it to numpy array
    return is_correct.cpu().numpy().tolist()      


def train(epoch_number, train_data_loader,model,loss_function, optimizer ):
    
    # define lists to contain the loss and accuracy values of each epoch
    losses, accuracies = [], []                      
    for epoch in range (epoch_number):               
        print(f"Epoch: {epoch+1}")
        #define lists to contain the loss and accuracy values of each batch
        batch_losses, batch_accuracies = [],[]       
        
        #create batches of training data by iterating thorugh data loader
        for batch in (iter(train_data_loader)):
            #unpack the batch
            input_batch, targets = batch
            #train the batch
            batch_loss = train_batch(input_batch, targets, model,loss_function, optimizer)  
            
            #store the loss value of each batch
            batch_losses.append(batch_loss)
            
        #get the mean of the loss values of all batches
        epoch_loss = np.array(batch_losses).mean()                        
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

def display_loss_accuracy(number_epoch,lossses, accuracies):
    
    epochs = np.arange(number_epoch)+1
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.title('Loss value over increasing epochs')
    plt.plot(epochs, lossses, label='Training Loss')
    plt.legend()
    plt.subplot(122)
    plt.title('Accuracy value over increasing epochs')
    plt.plot(epochs, accuracies, label='Training Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.legend()
    
    
def get_data(batchsize,train_images,train_targets,validation_images,validation_targets ):
    
    train_data = FMNISTDataset(train_images,train_targets) 
    train_data_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True) 
    validation_data = FMNISTDataset(validation_images,validation_targets) 
    validation_data_loader = DataLoader(validation_data, batch_size=len(validation_images), shuffle=False) 
    
    return train_data_loader, validation_data_loader


@torch.no_grad()
#define a function to calculate the loss value of the validation dataset
def get_validation_loss_value(input_batch, targets,model, loss_function):
    
    model.eval()
    predictions = model(input_batch)
    validation_loss = loss_function(predictions,targets)
    
    return validation_loss.item()


def train_with_validation(epoch_number,train_data_loader,validation_data_loader,model,loss_function, optimizer):
    
    train_losses, train_accuracies = [], []   
    validation_losses, validation_accuracies = [], [] 
    for epoch in range (epoch_number):               
        print(f"Epoch: {epoch+1}")
        batch_losses, batch_accuracies = [],[]       

        for batch in (iter(train_data_loader)):      
            input_batch, targets = batch                     
            batch_loss = train_batch(input_batch, targets, model,loss_function, optimizer)  

            batch_losses.append(batch_loss)                               
        epoch_loss = np.array(batch_losses).mean()                        
        train_losses.append(epoch_loss)
        print(f"Train Loss: {epoch_loss:0.3f}")

        for batch in (iter(train_data_loader)):
            input_batch, targets = batch
            is_correct = get_accuracy(input_batch, targets, model)

            batch_accuracies.extend(is_correct)
        epoch_accuracy = np.mean(batch_accuracies)
        train_accuracies.append(epoch_accuracy)
        print(f"Train Accuracy: {epoch_accuracy*100:0.0f}%")
        
        for batch in (iter(validation_data_loader)):
            input_batch, targets = batch
            validation_loss_value = get_validation_loss_value(input_batch, targets,model, loss_function)
            validation_accuracy = get_accuracy(input_batch, targets, model)
            
        validation_losses.append(validation_loss_value)
        print(f"Validation Loss: {validation_loss_value:0.3f}")
        valiation_epoch_accuracy = np.mean(validation_accuracy)
        print(f"Validation Accuracy: {valiation_epoch_accuracy*100:0.0f}%")
        validation_accuracies.append(valiation_epoch_accuracy)
        print('<--------------------------------------------------------->')
        
    return train_losses, train_accuracies, validation_losses, validation_accuracies


def display_train_validation_results(number_epoch,train_losses, train_accuracies, validation_losses, validation_accuracies):
    
    epochs = np.arange(number_epoch)+1
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.title('Training and Validation Loss value over increasing epochs')
    plt.plot(epochs, train_losses,'b', label='Training Loss')
    plt.plot(epochs, validation_losses,'r', label='Validation Loss')
    plt.legend()
    plt.subplot(122)
    plt.title('Training and Validation Accuracy value over increasing epochs')
    plt.plot(epochs, train_accuracies,'b', label='Training Accuracy')
    plt.plot(epochs, validation_accuracies,'r', label='Validation Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.legend()