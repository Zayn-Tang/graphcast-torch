import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import xarray
import data_factory.graph_tools as gg


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def load(self, scaler_dir):
        with open(scaler_dir, "rb") as f:
            pkl = pickle.load(f)
            self.mean = pkl["mean"]
            self.std = pkl["std"]

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class ERA5(torch.utils.data.Dataset):
    def __init__(self, ds, input_window_size=2, output_window_size=20) -> None:
        super().__init__()
        self.ds = ds

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        
        times = ds.time.values
        self.init_times = times[slice(input_window_size-1, -output_window_size-1)] 

    def __len__(self):
        return self.init_times.shape[0]
    
    def __getitem__(self, idx):
        assert idx < len(self.init_times)
        t = self.init_times[idx]
        t1 = t - pd.Timedelta(days=self.input_window_size//4, hours=(self.input_window_size % 4 - 1) * 6)

        t2 = t + pd.Timedelta(hours=6)
        t3 = t2 + pd.Timedelta(days=self.output_window_size//4, hours=(self.output_window_size % 4 - 1) * 6)

        tid = pd.date_range(t1, t, freq='6h')
        tid2 = pd.date_range(t2, t3, freq='6h')

        input = self.ds.sel(time=tid) # you can use subset of input, eg: only surface
        target = self.ds.sel(time=tid2)

        input = torch.from_numpy(input.values)
        target = torch.from_numpy(target.values)
        
        input = torch.nan_to_num(input) # t c h w 
        target = torch.nan_to_num(target) # t c h w

        return input, target



class EarthGraph(object):
    def __init__(self):
        self.mesh_data = None
        self.grid2mesh_data = None
        self.mesh2grid_data = None

    def generate_graph(self):
        mesh_nodes = gg.fetch_mesh_nodes()

        # mesh_6_edges, mesh_6_edges_attrs = gg.fetch_mesh_edges(6)
        # mesh_5_edges, mesh_5_edges_attrs = gg.fetch_mesh_edges(5)
        mesh_4_edges, mesh_4_edges_attrs = gg.fetch_mesh_edges(4)
        mesh_3_edges, mesh_3_edges_attrs = gg.fetch_mesh_edges(3)
        mesh_2_edges, mesh_2_edges_attrs = gg.fetch_mesh_edges(2)
        mesh_1_edges, mesh_1_edges_attrs = gg.fetch_mesh_edges(1)
        mesh_0_edges, mesh_0_edges_attrs = gg.fetch_mesh_edges(0)

        # mesh_edges = mesh_5_edges + mesh_4_edges + mesh_3_edges + mesh_2_edges + mesh_1_edges + mesh_0_edges
        mesh_edges = mesh_4_edges + mesh_3_edges + mesh_2_edges + mesh_1_edges + mesh_0_edges
        # mesh_edges_attrs = mesh_5_edges_attrs + mesh_4_edges_attrs + mesh_3_edges_attrs + mesh_2_edges_attrs + mesh_1_edges_attrs + mesh_0_edges_attrs
        mesh_edges_attrs = mesh_4_edges_attrs + mesh_3_edges_attrs + mesh_2_edges_attrs + mesh_1_edges_attrs + mesh_0_edges_attrs

        self.mesh_data = Data(x=torch.tensor(mesh_nodes, dtype=torch.float),
                              edge_index=torch.tensor(mesh_edges, dtype=torch.long).T.contiguous(),
                              edge_attr=torch.tensor(mesh_edges_attrs, dtype=torch.float))

        grid2mesh_edges, grid2mesh_edge_attrs = gg.fetch_grid2mesh_edges()
        self.grid2mesh_data = Data(x=None,
                                   edge_index=torch.tensor(grid2mesh_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(grid2mesh_edge_attrs, dtype=torch.float))

        mesh2grid_edges, mesh2grid_edge_attrs = gg.fetch_mesh2grid_edges()
        self.mesh2grid_data = Data(x=None,
                                   edge_index=torch.tensor(mesh2grid_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(mesh2grid_edge_attrs, dtype=torch.float))