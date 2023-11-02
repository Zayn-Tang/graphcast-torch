import argparse



def get_graphcast_args():
    parser = argparse.ArgumentParser('Graphcast training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('-g', '--gpuid', default=0, type=int,
                        help="which gpu to use")
    parser.add_argument("--world_size", default=3, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("--master_port", default="12355", type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    # Model parameters
    parser.add_argument('--grid-node-num', default=161 * 161, type=int, help='The number of grid nodes')
    parser.add_argument('--mesh-node-num', default=36 * 36, type=int, help='The number of mesh nodes')
    # parser.add_argument('--mesh-edge-num', default=2170, type=int, help='The number of mesh nodes')
    # parser.add_argument('--grid2mesh-edge-num', default=13570, type=int, help='The number of mesh nodes')
    # parser.add_argument('--mesh2grid-edge-num', default=13570, type=int, help='The number of mesh nodes')
    parser.add_argument('--grid-node-dim', default=70*2+2, type=int, help='The input dim of grid nodes')
    parser.add_argument('--grid-node-pred-dim', default=70, type=int, help='The output dim of grid-node prediction')
    parser.add_argument('--grid-node-infer-dim', default=5, type=int, help='The output dim of grid-node inference')
    parser.add_argument('--mesh-node-dim', default=3, type=int, help='The input dim of mesh nodes')
    parser.add_argument('--edge-dim', default=4, type=int, help='The input dim of all edges')
    parser.add_argument('--grid-node-embed-dim', default=256, type=int, help='The embedding dim of grid nodes')
    parser.add_argument('--mesh-node-embed-dim', default=128, type=int, help='The embedding dim of mesh nodes')
    parser.add_argument('--edge-embed-dim', default=128, type=int, help='The embedding dim of mesh nodes')
    parser.add_argument('--predict-steps', default=20, type=int, help='predict steps')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # Pipline training parameters
    parser.add_argument('--pp_size', type=int, default=8, help='pipeline parallel size')
    parser.add_argument('--chunks', type=int, default=1, help='chunk size')

    return parser.parse_args()