import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm

from metrics import class_error

log = logging.getLogger(name="image_trainer")


class sketch():
    def __init__(self, name):
        self.name = name
        self.format = format
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        
    def average(self):
        return self.sum / self.count
    
    def __str__(self):
       return f"{self.name}: {self.average()}"


def train_map(model, data_loader, device, epochs, lr, momentum, weight_decay, milestones, gamma):
    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    for epoch in range(epochs):
        epoch_loss = train( model, data_loader, criterion, optimizer, device)
        scheduler.step()
        log.info(f"Epoch {epoch} loss: {epoch_loss}")
    return model


def train(model, data_loader, criterion, optimizer, device):
    model.train()
    losses = sketch("Loss")
    for i, (images, target) in enumerate(tqdm(data_loader)):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.average()

def validate_map(model, data_loader, device):
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    losses = sketch("Loss")
    accuracies = sketch("Accuracy")
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)  
            acc = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))

    return losses.average(), accuracies.average()
    



def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=False)
    acc = pred.eq(target.data).sum().item() / target.shape[0]
    return acc