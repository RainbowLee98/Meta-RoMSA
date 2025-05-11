import argparse
import gc
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import pynvml
import torch


from config.config_regression import ConfigRegression
from data.load_data import MMDataLoader_meta
from details.email_message import send
from models.AMIO import AMIO
from trains.ATIO import ATIO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
current_time = time.strftime('%m%d%H%M%S')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args, dataloader):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    suffix = f'-mr{args.missing_rate[0]:.1f}-{args.seed}' if args.save_model else ''
    args.model_save_path = os.path.join(args.model_save_dir,
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}{suffix}-{current_time}.pth')
    # indicate used gpu
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # add tmp tensor to increase the temporary consumption of GPU
    tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load models
    # dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    del tmp_tensor

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)

    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    # do test
    results = atio.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    return results


def run_normal(args):
    res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds

    missing_rate = 0.0
    args = init_args
    # load config
    config = ConfigRegression(args)
    args = config.get_config()
    # load data
    dataloader = MMDataLoader_meta(args)
    # run results
    for i, seed in enumerate(seeds):
        if i == 0 and args.data_missing:
            missing_rate = str(round(args.missing_rate[0], 1))

        setup_seed(seed)
        args.seed = seed
        # missing_rate str类型 args.seed int类型
        # if missing_rate == '0.4' and args.seed ==111:
        #     print('跳过missing rate ==0.4 且seed==111情况')
        #     continue
        logger.info('Start running %s... with missing_rate=%s' % (args.modelName, missing_rate))
        logger.info(args)
        # runnning
        args.cur_time = i + 1
        test_results = run(args, dataloader)
        # restore results
        model_results.append(test_results)
        logger.info(f"==> Test results of seed {seed}:\n{test_results}")
    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(res_save_dir, \
                             f'{args.datasetName}-{args.train_mode}-{missing_rate}-{current_time}.csv')
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' % (save_path))
    # store results
    returned_res = res[1:]

    # detailed results
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(res_save_dir, \
                             f'{args.datasetName}-{args.train_mode}-{missing_rate}-detail-{current_time}.csv')
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Time", "Model", "Params", "Seed"] + criterions)
    # seed
    for i, seed in enumerate(seeds):
        res = [cur_time, args.modelName, str(args), f'{seed}']
        for c in criterions:
            val = round(model_results[i][c] * 100, 2)
            res.append(val)
        df.loc[len(df)] = res
    # mean
    res = [cur_time, args.modelName, str(args), '<mean/std>']
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    # max
    res = [cur_time, args.modelName, str(args), '<max/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        max_val = round(np.max(values) * 100, 2)
        max_seed = seeds[np.argmax(values)]
        res.append((max_val, max_seed))
    df.loc[len(df)] = res
    # min
    res = [cur_time, args.modelName, str(args), '<min/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        min_val = round(np.min(values) * 100, 2)
        min_seed = seeds[np.argmin(values)]
        res.append((min_val, min_seed))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Detailed results are added to %s...' % (save_path))

    return returned_res, criterions


def set_log(args):
    res_dir = os.path.join(args.res_save_dir, 'normals')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    log_file_path = os.path.join(res_dir, f'run-once-{args.modelName}-{args.datasetName}-{current_time}.log')
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    # 元学习
    parser.add_argument('--init_type', type=str, default='kaiming',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.012,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay when training')
    parser.add_argument('--alpha_lr', type=float, default=0.0001, help='initial learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of optimizer')
    parser.add_argument('--loss_meta_weight', type=float, default=0.3, help='Meta-learning loss weight')
    parser.add_argument('--support_loss_meta_weight', type=float, default=1.0, help='Meta-learning loss weight in support train')
    parser.add_argument('--loss_meta_weight_unimodal', type=float, default=0.5, help='The weights of audio and video modalities')
    parser.add_argument('--fusion_method', default='tensor', type=str, choices=['concat', 'add', 'mul', 'tensor'])
    parser.add_argument('--inner_lr', type=float, default=0.005, help='initial learning rate for optimizer')
    parser.add_argument('--uniweight_lr', type=float, default=0.0005, help='initial learning rate for optimizer')

    # 翻译器相关参数
    parser.add_argument('--embd_size', default=128, type=int, help='model embedding size')
    parser.add_argument('--mid_dim', type=int, default=512, help='Intermediate dimension')
    # parser.add_argument('--input_dim_t', type=int, default=512, help='lexical input dim')
    # parser.add_argument('--input_dim_a', type=int, default=512, help='lexical input dim')
    # parser.add_argument('--input_dim_v', type=int, default=512, help='lexical input dim')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
    # 翻译
    parser.add_argument('--trans_txt', type=float, default=1.0, help='The weight of text modal loop consistency')
    parser.add_argument('--trans_xtx', type=float, default=0.05,
                        help='The weight of audio/video modal loop consistency')
    parser.add_argument('--trans_xxx', type=float, default=0.05,
                        help='The weight of audio/video modal loop consistency')
    parser.add_argument('--trans_tx', type=float, default=4.0,
                        help='Weights that translate from text modal to other modal')
    parser.add_argument('--trans_xt', type=float, default=0.1,
                        help='Weights that translate from other modal to text modal')
    parser.add_argument('--trans_xx', type=float, default=0.05,
                        help='Weights that translate from other modal to text modal')

    # 常规
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='meta-romsa',
                        help='support meta-romsa')
    parser.add_argument('--datasetName', type=str, default='meta-simsv2',
                        help='support meta-sims/meta-mosi/meta-mosei/meta-simsv2')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--missing_rates', type=float, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=1111, help='start seed')
    parser.add_argument('--num_seeds', type=int, default=None, help='number of total seeds')

    parser.add_argument('--KeyEval', type=str, default='Loss', help='the evaluation metric used to select best model')
    parser.add_argument('--save_model', action='store_true', help='save the best model in each run (i.e., each seed)')

    # for sims
    parser.add_argument('--use_normalized_data', type=str, default='true',
                        help='use normalized audio & video data (for now, only for sims)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    global logger;
    logger = set_log(args)
    args.seeds = [111, 1111, 11111] if args.num_seeds is None else list(range(args.seed, args.seed + args.num_seeds))
    args.num_seeds = len(args.seeds)

    if args.missing_rates is None:
        if args.datasetName in ['meta-mosi', 'meta-mosei']:
            args.missing_rates = np.arange(0, 1.0 + 0.1, 0.1).round(1)
        else:
            args.missing_rates = np.arange(0, 0.5 + 0.1, 0.1).round(1)

    aggregated_results, metrics = [], []
    for mr in args.missing_rates:
        args.missing_rate = tuple([mr, mr, mr])
        res, criterions = run_normal(args)
        aggregated_results.append(res)
        metrics = criterions

    # save aggregated results
    save_path = os.path.join(args.res_save_dir, 'normals',
                             f'{args.datasetName}-{args.train_mode}-aggregated-{current_time}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model", "Missing_Rate"] + metrics)
    for mr, res in zip(args.missing_rates, aggregated_results):
        line = [args.modelName, mr] + res
        df.loc[len(df)] = line
    # auc
    agg_results = np.array(aggregated_results)[:, :, 0]
    auc_res = np.sum(agg_results[:-1] + agg_results[1:], axis=0) / 2 * 0.1
    df.loc[len(df)] = [args.modelName, 'AUC'] + auc_res.round(1).tolist()
    df.to_csv(save_path, index=None)
    logger.info('Aggregated results are added to %s...' % (save_path))

    send(args.modelName)
