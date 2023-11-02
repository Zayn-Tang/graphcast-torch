import numpy as np
import datetime

grid_lat = 161
grid_lon = 161
mesh_lat = 36
mesh_lon = 36

def id2position(node_id, lat_len, lon_len):
    lat = node_id // lon_len
    lon = node_id % lon_len
    cos_lat = np.cos((5 / 4 - (lat + 1) / (lat_len + 1)) * 2 * np.pi / 9)
    sin_lon = np.sin(5 * np.pi / 9 + lon * 2 /( lon_len * 9) * np.pi)
    cos_lon = np.cos(5 * np.pi / 9 + lon * 2 /( lon_len * 9) * np.pi)
    return cos_lat, sin_lon, cos_lon


def fetch_mesh_nodes():
    nodes = []
    for i in range(mesh_lat):
        # cos_lat = np.cos((0.5 - (i + 1) / (mesh_lat+1)) * np.pi)
        cos_lat = np.cos((5 / 4 - (i + 1) / (mesh_lat + 1)) * 2 * np.pi / 9)
        for j in range(mesh_lon):
            # sin_lon = np.sin(j / mesh_lon * np.pi)
            # cos_lon = np.cos(j / mesh_lon * np.pi)
            sin_lon = np.sin(5 * np.pi / 9 + j * 2 /( mesh_lon * 9) * np.pi)
            cos_lon = np.cos(5 * np.pi / 9 + j * 2 /( mesh_lon * 9) * np.pi)
            nodes.append([cos_lat, sin_lon, cos_lon])
    return nodes


def fetch_mesh_edges(r):
    """
    原代码中，128和320是mesh的个数，全部grid结点个数为720和1440
    修改后，30和36是mesh的个数（暂时），全部grid结点个数为161*161，分辨率为0.25，grid结点个数为644和644
    """
    assert 4 >= r >= 0

    step = 2 ** (4 - r)
    edges = []
    edge_attrs = []
    for i in range(0, mesh_lat, step):
        for j in range(0, mesh_lon, step):
            cur_node = id2position(i * mesh_lon + j, mesh_lat, mesh_lon)
            # 判断相距step步后的结点是否与当前结点相邻，
            # 下面是判断他的上下左右step步结点是否合法，添加相邻结点
            # tmp_attr 是计算两个结点直接position关系
            # 将得到的position关系和 tmp_attr 拼接得到边的属性
            if i - step >= 0:
                edges.append([(i - step) * mesh_lon + j, i * mesh_lon + j])
                target_node = id2position((i - step) * mesh_lon + j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            if i + step < mesh_lat:
                edges.append([(i + step) * mesh_lon + j, i * mesh_lon + j])
                target_node = id2position((i + step) * mesh_lon + j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            if j - step >= 0:
                edges.append([i * mesh_lon + j - step, i * mesh_lon + j])
                target_node = id2position(i * mesh_lon + j - step, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            '''
            else:
                edges.append([i * mesh_lon + mesh_lon - step, i * mesh_lon + j])
                target_node = id2position(i * mesh_lon + mesh_lon - step, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
             '''
            if j + step < mesh_lon:
                edges.append([i * mesh_lon + j + step, i * mesh_lon + j])
                target_node = id2position(i * mesh_lon + j + step, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            '''
            else:
                edges.append([i * mesh_lon, i * mesh_lon + j])
                target_node = id2position(i * mesh_lon, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            '''

    return edges, edge_attrs


def fetch_grid2mesh_edges():
    lat_span = grid_lat / mesh_lat
    lon_span = grid_lon / mesh_lon
    edges = []
    edge_attrs = []
    for i in range(grid_lat):
        for j in range(grid_lon):
            target_mesh_i = int(i / lat_span)
            target_mesh_j = int(j / lon_span)
            edges.append([i * grid_lon + j, target_mesh_i * mesh_lon + target_mesh_j])
            cur_node = id2position(i * grid_lon + j, grid_lat, grid_lon)
            target_node = id2position(target_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
            tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
            edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span - 0.1)# 5.6 若i/lat_span = 4, over_mesh_i = 3, target_mesh_i = 4
            if i / lat_span - 0.1 > 0 and over_mesh_i != target_mesh_i:
                edges.append([i * grid_lon + j, over_mesh_i * mesh_lon + target_mesh_j])
                target_node = id2position(over_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span + 0.1)
            if i / lat_span + 0.1 < mesh_lat and over_mesh_i != target_mesh_i:
                edges.append([i * grid_lon + j, over_mesh_i * mesh_lon + target_mesh_j])
                target_node = id2position(over_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span - 0.1)# 4.5
            '''
            if j / lon_span - 0.1 < 0:
                edges.append([i * grid_lon + grid_lon-1, target_mesh_i * mesh_lon + target_mesh_j])
                target_node = id2position(target_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            '''
            if j / lon_span - 0.1 > 0 and over_mesh_j != target_mesh_j:
                edges.append([i * grid_lon + j, target_mesh_i * mesh_lon + over_mesh_j])
                target_node = id2position(target_mesh_i * mesh_lon + over_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span + 0.1)
            '''
            if j / lon_span + 0.1 > mesh_lon:
                edges.append([i * grid_lon, target_mesh_i * mesh_lon + target_mesh_j])
                target_node = id2position(target_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            '''
            if j / lon_span + 0.1 < mesh_lon and over_mesh_j != target_mesh_j:
                edges.append([i * grid_lon + j, target_mesh_i * mesh_lon + over_mesh_j])
                target_node = id2position(target_mesh_i * mesh_lon + over_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

    return edges, edge_attrs


def fetch_mesh2grid_edges():
    lat_span = grid_lat / mesh_lat
    lon_span = grid_lon / mesh_lon
    edges = []
    edge_attrs = []
    for i in range(grid_lat):
        for j in range(grid_lon):
            target_mesh_i = int(i / lat_span)
            target_mesh_j = int(j / lon_span)
            edges.append([target_mesh_i * mesh_lon + target_mesh_j, i * grid_lon + j])
            target_node = id2position(i * grid_lon + j, grid_lat, grid_lon)
            cur_node = id2position(target_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
            tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
            edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span - 0.32)
            if i / lat_span - 0.32 > 0 and over_mesh_i != target_mesh_i:
                edges.append([over_mesh_i * mesh_lon + target_mesh_j, i * grid_lon + j])
                cur_node = id2position(over_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span + 0.32)
            if i / lat_span + 0.32 < mesh_lat and over_mesh_i != target_mesh_i:
                edges.append([over_mesh_i * mesh_lon + target_mesh_j, i * grid_lon + j])
                cur_node = id2position(over_mesh_i * mesh_lon + target_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span - 0.32)
            '''
            if j / lon_span - 0.3 < 0:
                edges.append([target_mesh_i * mesh_lon + (mesh_lon-1), i * grid_lon + j])
                cur_node = id2position(target_mesh_i * mesh_lon + (mesh_lon-1), mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            '''
            if j / lon_span - 0.32 > 0 and over_mesh_j != target_mesh_j:
                edges.append([target_mesh_i * mesh_lon + over_mesh_j, i * grid_lon + j])
                cur_node = id2position(target_mesh_i * mesh_lon + over_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span + 0.32)
            '''
            if j / lon_span + 0.3 > mesh_lon:
                edges.append([target_mesh_i * mesh_lon, i * grid_lon + j])
                cur_node = id2position(target_mesh_i * mesh_lon, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            '''
            if j / lon_span + 0.32 < mesh_lon and over_mesh_j != target_mesh_j:
                edges.append([target_mesh_i * mesh_lon + over_mesh_j, i * grid_lon + j])
                cur_node = id2position(target_mesh_i * mesh_lon + over_mesh_j, mesh_lat, mesh_lon)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

    return edges, edge_attrs


## 特征抽取
def fetch_time_features(cursor_time):

    year_hours = (datetime.date(cursor_time.year + 1, 1, 1) - datetime.date(cursor_time.year, 1, 1)).days * 24
    next_year_hours = (datetime.date(cursor_time.year + 2, 1, 1) - datetime.date(cursor_time.year + 1, 1, 1)).days * 24

    cur_hour = (cursor_time - datetime.datetime(cursor_time.year, 1, 1)) / datetime.timedelta(hours=1)
    time_features = []
    for j in range(grid_lon):
        # local time
        local_hour = cur_hour + j * 24 / grid_lon
        if local_hour > year_hours:
            tr = (local_hour - year_hours) / next_year_hours
        else:
            tr = local_hour / year_hours

        time_features.append([[np.sin(2 * np.pi * tr), np.cos(2 * np.pi * tr)]] * grid_lat)

    return np.transpose(np.asarray(time_features), [1, 0, 2])


def fetch_constant_features():
    constant_features = []
    for i in range(grid_lat):
        tmp = []
        for j in range(grid_lon):
            tmp.append(id2position(i * grid_lon + j, grid_lat, grid_lon))
        constant_features.append(tmp)
    return np.asarray(constant_features)



