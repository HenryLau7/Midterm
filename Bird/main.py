from utils import *
import torch
from dataset import get_dataloaders
from model import create_resnet34,create_resnet18
from optimizer import opt_ft
from train import trainer,test
import pytz
import os
import datetime
from dataset import get_dataloaders
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


def main(args):
    print(f"Pretrained={args.pretrained}, Training with lr={args.learning_rate}, decay={args.weight_decay},dropout = {args.dropout}, cutout = {args.cutout}, model = {args.model}")
    
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')    
    
    # time
    china_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.datetime.now(china_tz)
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)
        
    log_dir = args.logpath + '{}_lr_{}_decay_{}_pretrained_{}_dropout_{}_cutout{}_model{}/'.format(args.date,args.learning_rate, args.weight_decay, args.pretrained, args.dropout,args.cutout,args.model)
    writer = SummaryWriter(log_dir + "log")    

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    train_loader, test_loader = get_dataloaders(root_dir=args.datapath,batch_size=args.batch_size)
    
    if args.model == 'ResNet18':
        model = create_resnet18(num_classes=args.num_classes,pretrained=args.pretrained,dropout=args.dropout)
    elif args.model == 'ResNet34':
        model = create_resnet34(num_classes=args.num_classes,pretrained=args.pretrained,dropout=args.dropout)
    model.to(device)
    
    opt = opt_ft(model,lr=args.learning_rate, weight_decay=args.weight_decay)
    schedular = CosineAnnealingLR(opt,T_max=100,eta_min=0)
    
    Trainer = trainer(model, opt, schedular, args.epoch, train_loader, test_loader, log_dir, writer, device)
    # Trainer = trainer(model, opt, schedular, args.epoch, train_loader, test_loader, log_dir, writer)
    Trainer.train()
    
    if args.plot:
        plot_metrics(Trainer, log_dir)
    
    checkpoint = torch.load(log_dir + 'best_model.pth')
    model.load_state_dict(checkpoint['parameters'])
    test(model,test_loader,device)
    # test(model,test_loader)
    
    return 0


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--datapath', default= './fashion/')
    # parser.add_argument('--logpath', default='./logs/')
    # parser.add_argument('--batch_size', type=int, default=512)
    # parser.add_argument('--epoch', type=int, default=500)
    # parser.add_argument('--learning_rate', type=float, default=0.5)
    # parser.add_argument('--seed', type=int, default=43)
    # parser.add_argument('--date',default='0325')
    # parser.add_argument('--plot',default=True)
    # parser.add_argument('--reg',type=float, default=3e-4)
    # parser.add_argument('--decay',type=float, default=0.9)
    
    # learning_rate = [3e-3,3e-4]
    # decay= [3e-3,3e-4]

    # pretrain = [True, False]
    # pretrain = True
    # for pretrained, lr, wd in itertools.product(pretrain,learning_rate, decay):
        
    # pretrained = [True,False]
    # lr = [0.0003,0.003]
    # wd = [0.0003,0.003]

    # for pretrain in pretrained:
        # for learning_rate in lr:
            # for weight_decay in wd:
    # param_combinations = [
    #     (True, True, True, 'ResNet34', 3e-4),
    #     (True, False, True, 'ResNet34', 3e-4),
    #     (True, True, False, 'ResNet34', 3e-4),
    # ]
    
    param_combinations = [
        (False, True, True, 'ResNet34', 3e-4),
        (True, True, True, 'ResNet18', 3e-4),
        (True, True, True, 'ResNet34', 3e-5)
    ]

    # 遍历所有参数组合并训练模型
    for params in param_combinations:
        pretrain, cutout, dropout, model_name, lr = params
        
        class args:
            datapath = './CUB_200_2011/images'
            logpath = './logs/'
            batch_size = 512
            epoch = 500
            date = '0530'
            plot = True
            num_classes = 200
            learning_rate = lr
            weight_decay = 3e-4
            pretrained = pretrain
            device_id = 1
            dropout = dropout
            cutout = cutout
            model = model_name
        
        print(args)
        main(args)

