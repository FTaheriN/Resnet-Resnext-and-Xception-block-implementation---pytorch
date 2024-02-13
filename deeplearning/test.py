from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils import accuracy
from utils import load_model

device = "cpu"

def test(model, ckpt_path, val_loader, learning_rate):
    accuracy_ = []
    pred_ = []
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loading model
    model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer)
    
    model.eval()
    
    with torch.no_grad():
        loop_val = tqdm(
            enumerate(val_loader, 1),
            total=len(val_loader),
            desc="val",
            position=0,
            leave=True,
        )
        for batch_idx, (images, labels) in loop_val:
            optimizer.zero_grad()
            images = images.to(device).float()
            labels = labels.to(device)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            acc1 = accuracy(labels_pred, labels)
            accuracy_.append(acc1)
            pred_ += labels_pred
    return accuracy_, pred_