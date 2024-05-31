import numpy as np
import torch
import torch.nn as nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class trainer:
    def __init__(self, model, optimizer,schedular, epoch, train_loader, vali_loader, save_dir, writer, device):
        self.model = model
        self.optimizer = optimizer
        self.schedular = schedular
        self.epoch = epoch
        self.epochs_per_vali = 5
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.criterion = nn.CrossEntropyLoss()
        self.writer = writer
        
        self.epochs_per_vali = 5
        
        self.best_acc = 0
        self.save_dir = save_dir
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.device = device
        
    def train(self):
        
        for epoch in range(self.epoch):
            total = 0
            correct = 0
            loss_meter = AverageMeter()
            
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                label = label.long()
                total += data.shape[0]
                output = self.model(data)
                                
                loss = self.criterion(output,label)
                prediction = torch.argmax(output, axis=1)
                correct += torch.sum(prediction == label).item()
                loss_meter.update(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            acc = 100.*correct/total
            self.train_losses.append(loss_meter.avg)
            self.train_accuracies.append(acc)
            self.schedular.step()
            
            print("Epoch:{}, Loss:{}, Acc:{}, LR:{}".format(epoch, loss_meter.avg, acc, self.optimizer.param_groups[1]['lr']))
            
            self.writer.add_scalars('loss', {'train_loss': loss_meter.avg}, epoch)
            self.writer.add_scalars('acc', {'train_acc': acc}, epoch)
            self.writer.add_scalars('LR', {'LR': self.optimizer.param_groups[1]['lr']}, epoch)

            
            if (epoch+1) % self.epochs_per_vali == 0:
                val_loss, val_acc = self.validation(epoch)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                
    def validation(self, epoch):
        total = 0
        correct = 0
        loss_meter = AverageMeter()
        with torch.no_grad():  # 禁用梯度计算
            for data, label in self.vali_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                total += data.shape[0]
                output = self.model(data)
                loss = self.criterion(output,label)
                
                prediction = torch.argmax(output, axis=1)
                correct += torch.sum(prediction == label).item()
                loss_meter.update(loss.item())
            
        acc = 100.*correct/total
        print("Validation: Loss:{}, Acc:{}".format(loss_meter.avg, acc))
        
        self.writer.add_scalars('loss', {'valid_loss': loss_meter.avg}, epoch)
        self.writer.add_scalars('acc', {'valid_acc': acc}, epoch)
        
        if acc > self.best_acc:
            self.best_acc = acc
            print('Best model updated!')
            checkpoints = {'parameters':self.model.state_dict(),
                           'acc':acc,
                           'epoch':epoch
            }
            torch.save(checkpoints,self.save_dir+'best_model.pth')
            
        return loss_meter.avg, acc
                


def test(model, test_loader,device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():  
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            total += data.shape[0]
            output = model(data)
            
            # prediction = torch.argmax(output, axis=1)
            # correct += torch.sum(prediction == label).item()  # 使用 .item() 获取 scalar value
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
        
        
    acc = 100.*correct/total
    print("Test: Acc:{}".format(acc))