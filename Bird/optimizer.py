import torch.optim as optim


def opt_ft(model, lr, weight_decay):
    pretrained_params = [param for name, param in model.named_parameters() if 'fc' not in name]
    new_params = model.fc.parameters()

    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': lr*0.1,  'weight_decay': weight_decay*0.1 },  # lr and weight_decay are both 0.1 times
        {'params': new_params, 'lr': lr, 'weight_decay': weight_decay}          
    ])
    return optimizer