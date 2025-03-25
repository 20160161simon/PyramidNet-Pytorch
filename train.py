from dataset import CIFAR10_4x
from evaluation import evaluation
from model import Net
from Tricks.cutout import Cutout

import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import time
from datetime import datetime, timedelta
import sys

from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser(description="PyramidNet-Bottleneck")
parser.add_argument("--depth", type=int, default=110, help="depth of PyramidNet")
parser.add_argument("--alpha", type=int, default=270, help="alpha of PyramidNet")
parser.add_argument("--data-root-dir", type=str, default=".")
parser.add_argument("--batch-size", type=int, default=32, help="batch size for each GPU")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--model-name", type=str, default="model")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--mixup-alpha", type=float, default=1.0)
parser.add_argument("--warmup-epochs", type=int, default=5)
parser.add_argument("--world-size", type=int, default=3, help="number of GPU you have")

def main(rank, world_size):
    global args
    args = parser.parse_args()

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    model = Net(depth=args.depth, alpha=args.alpha).to(rank)
    model = DDP(model, device_ids=[rank])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(128, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([125/255, 124/255, 115/255], 
                            [60/255, 59/255, 64/255]),
        Cutout(64, prob=0.5)
    ])

    trainset = CIFAR10_4x(root=args.data_root_dir, split="train", transform=train_transform)
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_dataloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([125 / 255, 124 / 255, 115 / 255],
                             [60 / 255, 59 / 255, 64 / 255])
    ])
    valset = CIFAR10_4x(root=args.data_root_dir, split="valid", transform=val_transform)
    val_sampler = DistributedSampler(
        valset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    val_dataloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # Learning Rate Scheduler (Warmup + CosineAnnealing)
    base_lr = args.lr
    warmup_start_lr = 0.01 * base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_start_lr

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs, 
        eta_min=0.
    )

    criterion = nn.CrossEntropyLoss().to(rank)
    
    if rank == 0:
        print("start training!")
        total_epochs = args.epochs
        best_accuracy = 0
        start_time = time.time()

    for epoch in range(args.epochs):
        
        if rank == 0:
            epoch_start_time = time.time()

        model.train()
        train_sampler.set_epoch(epoch)

        if epoch < args.warmup_epochs:
            lr_scale = (base_lr - warmup_start_lr) / args.warmup_epochs
            current_lr = warmup_start_lr + epoch * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            if args.mixup_alpha > 0:
                mixed_inputs, y_a, y_b, lam = mixup_data(
                    inputs, targets, args.mixup_alpha, rank
                )
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            if args.mixup_alpha > 0:
                total_correct += (lam * preds.eq(y_a).sum().item() + 
                                (1 - lam) * preds.eq(y_b).sum().item())
            else:
                total_correct += preds.eq(targets).sum().item()
            total_samples += batch_size


        # Synchronize metrics across all processes
        total_loss_tensor = torch.tensor(total_loss, dtype=torch.float64, device=rank)
        total_correct_tensor = torch.tensor(total_correct, dtype=torch.float64, device=rank)
        total_samples_tensor = torch.tensor(total_samples, dtype=torch.float64, device=rank)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        accuracy = total_correct_tensor.item() / total_samples_tensor.item()
        
        avg_val_loss, val_accuracy = validate(model, criterion, val_dataloader, rank)
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start_time
            
            elapsed_time = time.time() - start_time 
            avg_time_per_epoch = elapsed_time / (epoch + 1) 
            remaining_time = avg_time_per_epoch * (total_epochs - epoch - 1)
            
            def format_time(seconds):
                return str(timedelta(seconds=int(seconds)))
            
            print(f'Epoch [{epoch+1}/{total_epochs}] | '
                f'LR: {current_lr:.6f} | '
                f'Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2%} | '
                f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2%} | '
                f'Epoch Time: {format_time(epoch_time)} | '
                f'Elapsed: {format_time(elapsed_time)} | '
                f'Remaining: {format_time(remaining_time)}')
            
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }
            model_dir = './models'
            torch.save(checkpoint, os.path.join(model_dir, f'{args.model_name}.pth'))

            if val_accuracy > best_accuracy:
                torch.save(checkpoint, os.path.join(model_dir, f'{args.model_name}-earlystop.pth'))
                sys.stderr.write(f"Early stop at epoch {epoch} !\n")
            
            best_accuracy = max(best_accuracy, val_accuracy)
        
        if epoch >= args.warmup_epochs:
            scheduler.step()

    dist.destroy_process_group()

def set_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > (2**32 - 1):
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        return x, y, y, 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def validate(model, criterion, val_dataloader, rank):
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for data, targets in val_dataloader:
            data, targets = data.to(rank), targets.to(rank)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            batch_size = targets.size(0)
            total_val_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total_val_correct += (predicted == targets).sum().item()
            total_val_samples += batch_size

    total_val_loss_tensor = torch.tensor(total_val_loss, dtype=torch.float64, device=rank)
    total_val_correct_tensor = torch.tensor(total_val_correct, dtype=torch.float64, device=rank)
    total_val_samples_tensor = torch.tensor(total_val_samples, dtype=torch.float64, device=rank)
    
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_val_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_val_samples_tensor, op=dist.ReduceOp.SUM)

    return (total_val_loss_tensor.item() / total_val_samples_tensor.item(), total_val_correct_tensor.item() / total_val_samples_tensor.item())

def print_para_num(model):
    print("number of trained parameters: %d" % (
        sum([param.nelement() for param in model.parameters() if param.requires_grad])))
    print("number of total parameters: %d" %
        (sum([param.nelement() for param in model.parameters()])))

if __name__ == '__main__':
    world_size = 3
    torch.multiprocessing.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )