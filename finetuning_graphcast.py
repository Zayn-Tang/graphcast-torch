import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import xarray as xr
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import timm.optim
from timm.scheduler import create_scheduler
from torch.cuda.amp import GradScaler
from data_factory.datasets import ERA5, EarthGraph
from model.graphcast_sequential import GraphCast
from utils.params import get_graphcast_args
from utils.tools import load_model, save_model
import pickle
# from utils.eval import graphcast_evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

'''
import torch
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
print(local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print(device)
'''

#--nproc_per_node=2 一般用几张卡,就设为几


SAVE_PATH = Path('./output/graphcast20/')
# SAVE_PATH.mkdir(parents=True, exist_ok=True)

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print(device)
data_dir = '/root/autodl-tmp'

def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds

def load_dataset():
    ds = []
    for y in range(2007, 2017):
        data_name = os.path.join(data_dir, f'weather_round1_train_{y}')
        x = xr.open_zarr(data_name, consolidated=True)
        print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')
        ds.append(x)
    ds = xr.concat(ds, 'time')
    ds = chunk_time(ds)
    return ds

# ds = load_dataset().x
# shape = ds.shape # batch x channel x lat x lon 
# times = ds.time.values

# init_times = times[slice(1, -21)] 
# num_data = len(init_times)
# names = list(ds.channel.values)
# test_names = names[-5:]

def compute_rmse(out, tgt):
    rmse = torch.sqrt(((out - tgt)**2).mean())
    return rmse

climates = {
    't2m': 3.1084048748016357,
    'u10': 4.114771819114685,
    'v10': 4.184110546112061,
    'msl': 729.5839385986328,
    'tp': 0.49046186606089276,
}

def run_eval(output, target):
    '''
        output: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
        target: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
    '''
    result = {}
    output = output.detach()
    target = target.detach()

    for cid, (name, clim) in enumerate(climates.items()):
        res = []
        for sid in range(output.shape[1]):
            out = output[:, sid, cid] # [N, H, W] 每个时间步的每个特征
            tgt = target[:, sid, cid]
            rmse = compute_rmse(out, tgt) # 
            rmse = reduce_tensor(rmse).item()
            # rmse = rmse.to(torch.device("cpu"))
            nrmse = (rmse - clim) / clim
            res.append(nrmse)

        score = max(0, -np.mean(res))
        result[name] = float(score)

    score = np.mean(list(result.values()))
    result['score'] = float(score) 
    return result

def run_eval_valid(output, target):
    '''
        output: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
        target: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
    '''
    result = {}
    output = output.detach()
    target = target.detach()

    for cid, (name, clim) in enumerate(climates.items()):
        res = []
        for sid in range(output.shape[1]):
            out = output[:, sid, cid] # [N, H, W] 每个时间步的每个特征
            tgt = target[:, sid, cid]
            rmse = compute_rmse(out, tgt) # 
            # rmse = reduce_tensor(rmse).item()
            rmse = rmse.to(torch.device("cpu"))
            nrmse = (rmse - clim) / clim
            res.append(nrmse)

        score = max(0, -np.mean(res))
        result[name] = float(score)

    score = np.mean(list(result.values()))
    result['score'] = float(score) 
    return result


def average_score(RMSE_list, key):
    score = 0
    for RMSE in RMSE_list:
        score += RMSE[key]
    return score/len(RMSE_list)


def train_one_epoch(epoch, model, criterion, data_loader, graph, optimizer, predict_steps, weight, lat_weight, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    score_all = torch.tensor(0.).to(device)
    model.train()

    input_x = [
        None,
        graph.mesh_data.x.half().cuda(non_blocking=True),
        graph.mesh_data.edge_index.cuda(non_blocking=True),
        graph.mesh_data.edge_attr.half().cuda(non_blocking=True),
        graph.grid2mesh_data.edge_index.cuda(non_blocking=True),
        graph.grid2mesh_data.edge_attr.half().cuda(non_blocking=True),
        graph.mesh2grid_data.edge_index.cuda(non_blocking=True),
        graph.mesh2grid_data.edge_attr.half().cuda(non_blocking=True)
    ]
    '''
    weight = get_weight(args) # [batch, channel, h, w]
    weight = weight.unsqueeze(1).to(device) # [batch, 1, channel, h, w]
    # diff_std = get_diff_std(args)
    # diff_std = diff_std.unsqueeze(1).to(device)# [batch, 1, channel, h, w]
    '''
    scaler = GradScaler()
    for step, batch in enumerate(data_loader):
        x, y = [x.half().cuda(non_blocking=True) for x in batch]
        y = y[:, :predict_steps, ...]
        input_x[0] = x
        bs,ts,c,h,w = x.shape
        # print(bs)

        pred_list = []
        optimizer.zero_grad()
        for t in range(predict_steps):
            # optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # out = model(input_x)
                out = model(*input_x)
                out = out.reshape(bs, h,w, c).permute(0, 3, 1, 2) # [bs, c, h, w]
            out = out.unsqueeze(1)
            pred_list.append(out + x[:, 1:, ...])  # [bs, 1, c, h, w]
            x = torch.concat([x[:,1:,...], x[:,1:,...]+out], dim=1)
            input_x[0] = x

        pred = torch.concat(pred_list,dim=1)
        loss = criterion(pred*weight*lat_weight, y*weight*lat_weight)
        # print(f'step {step}, loss:{loss}')
        # loss_all += loss
        loss_all += reduce_tensor(loss).item()#有多个进程，把进程0和1的loss加起来平均
        # print(f'step {step}, loss:{loss_all}')
        count += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # RMSE 
        score = run_eval(pred[...,-5:,30:-30,30:-30], y[...,-5:,30:-30,30:-30])
        score_all += score["score"]
        #score_all += reduce_tensor(torch.tensor(score["score"]).to(device)).item()

        if step % 200 == 0 and device==0:
            print("Step: ", step, " | Training Aver Loss:", (loss_all/count).item(), " | Train Eval Score: ", (score_all/count).item(), flush=True)
        
        if step == 1000:
            # save_model(model, step, optimizer=optimizer, lr_scheduler=None, min_loss=min_loss, path= SAVE_PATH / 'latest.pt')
            return loss_all/count

    return loss_all/count


@torch.no_grad()
def graphcast_evaluate(data_loader, graph, model, criterion, predict_steps, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    score_all = torch.tensor(0.).to(device)

    input_x = [
        None, # gx
        graph.mesh_data.x.half().cuda(non_blocking=True), #mx
        graph.mesh_data.edge_index.cuda(non_blocking=True), # me_i
        graph.mesh_data.edge_attr.half().cuda(non_blocking=True), # me_x
        graph.grid2mesh_data.edge_index.cuda(non_blocking=True), # g2me_i
        graph.grid2mesh_data.edge_attr.half().cuda(non_blocking=True), # g2me_x
        graph.mesh2grid_data.edge_index.cuda(non_blocking=True),# m2ge_i
        graph.mesh2grid_data.edge_attr.half().cuda(non_blocking=True) # m2ge_x
    ]

    # switch to evaluation mode
    model.eval()
    for step, batch in enumerate(data_loader):
        pred_list = []
        x, y = [x.half().cuda(non_blocking=True) for x in batch]
        y = y[:, :predict_steps, ...]
        input_x[0] = x
        bs,ts,c,h,w = x.shape
        # y [batch, time(20), channel(70), h, w]

        for t in range(predict_steps):
            with torch.cuda.amp.autocast():
                out = model(*input_x)
                out = out.reshape(bs, h, w, c).permute(0, 3, 1, 2) # [bs, c, h, w]
            out = out.unsqueeze(1)
            pred_list.append(out + x[:, 1:, ...])  # [bs, 1, c, h, w]
            x = torch.concat([x[:,1:,...], x[:,1:,...]+out], dim=1)
            input_x[0] = x
        
        pred = torch.concat(pred_list,dim=1)
        loss = criterion(pred[:,:,-5:,...], y[:,:,-5:,...])
        loss_all += loss.item()
        # loss_all += reduce_tensor(loss)#有多个进程，把进程0和1的loss加起来平均
        count += 1

        score = run_eval_valid(pred[...,-5:,30:-30,30:-30], y[...,-5:,30:-30,30:-30])
        score_all += score["score"]
        # score_all += reduce_tensor(torch.tensor(score["score"]).to(device)).item()
        
        if step % 200 == 0 and device==0:
            print("Step: ", step, " | Valid Aver Loss:", (loss_all/count).item(), " | Valid Eval Score: ", (score_all/count).item(), flush=True)

    return loss_all / count, score_all/count

def get_weight(args):
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    k = 1 / np.sum(levels)

    weight = torch.zeros(13)
    for i, level in enumerate(levels):
        weight[i] = level*k
    weight = torch.concat([weight.repeat(5), torch.ones(5)])
    weight = torch.sqrt(weight)
    weight = weight.reshape(1, 70, 1, 1).repeat(args.batch_size, 1, 161, 161)
    return weight

def get_lat_lon_weight(args):
    weight = torch.ones(args.grid_node_num).reshape(161, 161)
    for i in range(30):
        if i == 0:
            weight[i, :] = 0.1 + i * 0.03
            weight[-(i+1), :] = 0.1 + i * 0.03
            weight[:, i] = 0.1 + i * 0.03
            weight[:, -(i+1)] = 0.1 + i * 0.03
        else:
            weight[i, i:-i] = 0.1 + i * 0.03
            weight[-(i+1), i:-i] = 0.1 + i * 0.03
            weight[i:-i, i] = 0.1 + i * 0.03
            weight[i:-i, -(i+1)] = 0.1 + i * 0.03
    weight = torch.sqrt(weight)
    weight = weight.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, 1, 1, 1)
    return weight

def get_lat_weight(lat, args):
    diff = np.diff(lat)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    delta_latitude = np.abs(diff[0])
    # print(delta_latitude)
    weights = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(delta_latitude/2))
    # print(weights)
    weights[0] = np.sin(np.deg2rad(delta_latitude/4)) * np.cos(np.deg2rad(50 - delta_latitude/4))
    weights[-1] = np.sin(np.deg2rad(delta_latitude/4)) * np.cos(np.deg2rad(10 + delta_latitude/4))
    # print(weights)
    weights = weights / weights.mean()
    weights = np.sqrt(weights)
    weights = torch.tensor(weights).reshape(1, 1, 1, -1, 1).repeat(args.batch_size, 1, 1, 1, 161)
    return weights

def get_step_weight(args):
    pretrain_steps = 0
    k = 1 / (args.predict_steps - pretrain_steps)
    weight = torch.ones(args.predict_steps)
    for i in range(pretrain_steps, args.predict_steps):
        weight[i] += i * k
    weight = torch.sqrt(weight)
    weight = weight.reshape(1, -1, 1, 1, 1).repeat(args.batch_size, 1, 1, 1, 1) # [bs, 20, 1, 1, 1]
    return weight
        
def get_diff_std(args):
    with open("./scaler.pkl", "rb") as f:
            pkl = pickle.load(f)
            channels = pkl["channels"]
            std_r = pkl["std"]
    
    std = torch.tensor(std_r)
    std = std.reshape(1, 70, 1, 1).repeat(args.batch_size, 1, 161, 161)
    return std

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  # 总进程数
    return rt
    


def train(local_rank, args, ds, num_data):
    '''
    Args:
        local_rank: 本地进程编号
        rank: 进程的global编号
        local_size: 每个节点几个进程
        word_size: 进程总数
    '''
    # 初始化
    rank = local_rank
    gpu = local_rank
    torch.cuda.set_device(gpu)
    dist.init_process_group("nccl",
                            #init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            init_method="tcp://localhost:22355",
                            rank=rank,
                            world_size=args.world_size)


    # generate graph
    if os.path.exists("./EarthGraph"):
        graph = torch.load("./EarthGraph")
    else:
        graph = EarthGraph()
        graph.generate_graph()
        torch.save(graph, "./EarthGraph")

    args.grid2mesh_edge_num = graph.grid2mesh_data.num_edges
    args.mesh2grid_edge_num = graph.mesh2grid_data.num_edges
    args.mesh_edge_num = graph.mesh_data.num_edges

    # 模型初始化
    model = GraphCast(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # lr_scheduler, _ = create_scheduler(args, optimizer) 删了
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.MSELoss()
    start_epoch, start_step, min_loss = load_model(model, optimizer, path=SAVE_PATH / 'pretrain08.pt', only_model=True)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    
    train_num_data = int(num_data * 0.9)
    valid_num_data = int(num_data * 0.1)
    train_ds_time = ds.time.values[slice(1, train_num_data)]
    valid_ds_time = ds.time.values[slice(train_num_data, train_num_data+valid_num_data)]
    train_ds = ds.sel(time=train_ds_time)
    valid_ds = ds.sel(time=valid_ds_time)

    # dataset 初始化
    train_dataset = ERA5(train_ds, output_window_size=args.predict_steps)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size,
                              sampler=train_sampler, drop_last=True, shuffle=False, num_workers=0, pin_memory = True)
    valid_dataset=ERA5(valid_ds, output_window_size=args.predict_steps)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=args.batch_size, 
                              drop_last=True, num_workers=0, pin_memory = True)


    # load
    # start_epoch, start_step, min_loss = load_model(model, optimizer, path=SAVE_PATH / 'latest.pt')
    max_score = 0.72
    # start_epoch = 10
    # min_loss = 100
    
    weight = get_weight(args) # [batch, channel, h, w]
    weight = weight.unsqueeze(1).to(gpu) # [batch, 1, channel, h, w]
    lat_lon_weight = get_lat_lon_weight(args)
    lat_lon_weight = lat_lon_weight.unsqueeze(1).to(gpu)
    # step_weight = get_step_weight(args).to(gpu) # [bs, 20, 1, 1, 1]
    lat_values = ds.lat.values
    lat_weight = get_lat_weight(lat_values, args).to(gpu)
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(epoch, model, criterion, train_loader, graph, optimizer, args.predict_steps, weight, lat_weight, device=gpu)
        # 删了
        # lr_scheduler.step(epoch)
        # save_model(model, epoch + 1, optimizer=optimizer, lr_scheduler=lr_scheduler, min_loss=min_loss, path= SAVE_PATH / 'latest.pt')

        # val_loss, val_score = graphcast_evaluate(valid_loader, graph, model, criterion, args.predict_steps, device=gpu)

        # print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train loss: {train_loss.item():.6f} | Val loss: {val_loss.item():.6f}, Val score: {val_score.item():.6f}")

        # save model
        if gpu == 0:
            val_loss, val_score = graphcast_evaluate(valid_loader, graph, model, criterion, args.predict_steps, device=gpu)
            
            print(f"Epoch {epoch+1} | LR: {optimizer.param_groups[0]['lr']:.7f} | Train loss: {train_loss.item():.6f} | Val loss: {val_loss.item():.6f}, Val score: {val_score.item():.6f}", flush=True)
            
            save_model(model, epoch + 1, optimizer=optimizer, lr_scheduler=None, min_loss=min_loss, path= SAVE_PATH / 'latest.pt')
            if val_score > max_score:
                max_score = val_score
                min_loss = val_loss
                save_model(model, path=SAVE_PATH / f'epoch{epoch+1}_{val_score:.6f}_best.pt', min_loss=min_loss, only_model=True)
        dist.barrier()
        # lr_scheduler.step(max_score)

if __name__=="__main__":
    args = get_graphcast_args()
    
    ds = load_dataset().x
    # shape = ds.shape # batch x channel x lat x lon 
    times = ds.time.values

    init_times = times[slice(1, -21)] 
    num_data = len(init_times)
    
    torch.manual_seed(2023)
    np.random.seed(2023)
    cudnn.benchmark = True
    # train(args.gpuid,args)
    mp.spawn(train, args=(args, ds, num_data), nprocs=3, join=True)







