#!/usr/bin/python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from natsort import natsorted
import logging
import random
from config import DefaultConfigure
from dataset import DataSet
from unet import UNet
from loss import MyLoss
from CSI import find_best_CSI
import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(opt):
    # seed
    setup_seed(1)
    # dataset
    train_dataset = DataSet(opt.train_data_root)
    logging.info('train dataset sample num:%d' % len(train_dataset))
    trainloader = DataLoader(train_dataset,
                             shuffle=True,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers)
    # network
    net = UNet(22, [32, 64, 128, 256])
    if opt.checkpoint_model:
        net.load_state_dict(torch.load(os.path.join(opt.load_model_path, opt.checkpoint_model)))
    if opt.use_gpu:
        net.cuda()

    # loss function
    loss_func = MyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    
    # optimizer
    optimizer = optim.Adam(net.parameters())
    iteration = 1
    if opt.optimizer:
        optimizer.load_state_dict(torch.load(os.path.join(opt.load_optimizer_path, opt.optimizer)))
        iteration = int(opt.optimizer.split('.')[0].split('_')[-1]) + 1

    # training
    while iteration <= opt.max_iter:
        for ii, (input, label) in enumerate(trainloader):
            if opt.use_gpu:
                input = input.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = net(input)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            # print loss
            if (ii+1) % opt.display == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                logging.info('iteration: %d, lr: %f, loss: %.6f' % (iteration, lr, loss))
                print('iteration: %d, lr: %f, loss: %.6f' % (iteration, lr, loss))
            # save model and optimizer
            if iteration % opt.snapshot == 0:
                torch.save(net.state_dict(),
						os.path.join(opt.load_model_path,
                                opt.model+'_'+str(iteration)+'.pth'))
                torch.save(optimizer.state_dict(),
						os.path.join(opt.load_optimizer_path,
                                'optim_'+str(iteration)+'.pth'))
                logging.info(opt.model + '_' + str(iteration) + '.pth saved')
            iteration += 1
            if iteration > opt.max_iter:
                break

def val(opt):
    # dataset
    val_dataset = DataSet(opt.val_data_root)
    valloader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers)
    # network
    net = UNet(22, [32, 64, 128, 256])
    net.eval()
    models = natsorted(os.listdir(opt.load_model_path))

    CSI = np.zeros((len(models), 5), dtype=float)
    CSI[:, 0] = np.arange(opt.snapshot, opt.max_iter+1, opt.snapshot)

    with torch.no_grad():
        for iteration, model in enumerate(models):
            print(model)
            logging.info(model)
            dec_value = []
            labels = []
            net.load_state_dict(torch.load(os.path.join(opt.load_model_path, model)))
            if opt.use_gpu:
                net.cuda()
            #  softmax output
            for input, target in valloader:
                if opt.use_gpu:
                    input = input.cuda()
                output = net(input).permute(0, 2, 3, 1).contiguous().view(-1, 2)
                target = target.view(-1)
                output = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
                dec_value.append(output)
                labels.append(target.numpy())

            dec_value = np.concatenate(dec_value)
            labels = np.concatenate(labels).squeeze()
            # save dec_value
            np.savetxt(os.path.join(opt.result_file,
                                    'iteration_' + str((iteration+1)*opt.snapshot) + '.txt'), dec_value, fmt='%10.6f')
            # find best CSI
            CSI[iteration, 1:] = find_best_CSI(dec_value, labels)
            # save CSI to file every epoch
            np.savetxt(opt.result_file + '/CSI.txt', CSI, fmt='%8d'+'%8.4f'*4)

    best_iteration = np.arange(opt.snapshot, opt.max_iter+1, opt.snapshot)[np.argmax(CSI[:,1])]
    confidence = CSI[int(best_iteration/opt.snapshot)-1, 4]
    logging.info('best_iteration: %d,confidence: %.6f' % (best_iteration, confidence))

    return best_iteration, confidence

def test(opt):
    # dataset
    test_dataset = DataSet(opt.test_data_root, test=True)
    testloader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers)

    # network
    net = UNet(22, [32, 64, 128, 256])
    net.eval()
    net.load_state_dict(torch.load(os.path.join(opt.load_model_path, opt.checkpoint_model)))
    if opt.use_gpu:
        net.cuda()

    dec_value = []
    labels = []
    
    with torch.no_grad():
        #  softmax output
        for input, target in testloader:
            if opt.use_gpu:
                input = input.cuda()
            output = net(input).permute(0, 2, 3, 1).contiguous().view(-1, 2)
            output = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
            target = target.view(-1)
            dec_value.append(output)
            labels.append(target.numpy())

    dec_value = np.concatenate(dec_value)
    labels = np.concatenate(labels).squeeze()
    # save dec_value
    np.savetxt(os.path.join(opt.result_file,
                            'best_iteration_' + str(opt.best_iteration) + '.txt'), dec_value, fmt='%10.6f')
    # find best CSI
    res = find_best_CSI(dec_value, labels, opt.confidence)
    print(res)
    np.savetxt(os.path.join(opt.result_file, 'test_result.txt'), [res],
               fmt='CSI:%.6f\nPOD:%.6f\nFAR:%.6f\nconfidence:%.6f')

if __name__ == '__main__':
    # set config
    opt = DefaultConfigure()
    # set log
    logging.basicConfig(filename=os.path.join(opt.result_file, opt.log_name), level=logging.INFO)
    # set CUDA environ
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
    # train
    print('training...')
    start_time = time.time()
    train(opt)
    logging.info('use time: %.6f' % (time.time() - start_time))
    print('train finish, now val...')
    # val
    start_time = time.time()
    opt.best_iteration, opt.confidence = val(opt)
    opt.checkpoint_model = opt.model + '_' + str(opt.best_iteration) + '.pth'
    logging.info('val use time: %.6f' % (time.time() - start_time))
    print('val finish, now test...')
    # test
    test(opt)
    print('finish')

