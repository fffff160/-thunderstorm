#!/usr/bin/python


class DefaultConfigure(object):
    # data path
    train_data_root = '/data/wind_train_data/'
    val_data_root = '/data/wind_val_data/'
    test_data_root = '/data/wind_test_data/'

    # checkpoint path
    model = 'unet'
    load_model_path = './checkpoints'
    checkpoint_model =  None  # 'xxx.pth'
    # optimizer_state
    load_optimizer_path = './optimizer_state'
    optimizer = None  # 'xxx.pth'

    use_gpu = True
    device = 0

    batch_size = 5
    num_workers = 4
    display = 100
    snapshot = 4000

    max_iter = 100000
    lr = 0.05
    momentum = 0.9
    weight_decay = 1e-4

    result_file = './result'
    log_name = 'train.log'
