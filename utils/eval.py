import torch


@torch.no_grad()
def fourcastnet_pretrain_evaluate(data_loader, model, criterion):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        _, x, y = [x.half().cuda(non_blocking=True) for x in batch]
        x = x.transpose(3, 2).transpose(2, 1)
        y = y.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(x)
            tmp_loss = criterion(out, y)
            if torch.isnan(tmp_loss).int().sum() == 0:
                count += 1
                loss += tmp_loss

    dist.reduce(loss, 0)
    dist.reduce(count, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def fourcastnet_finetune_evaluate(data_loader, model, criterion):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        xt0, xt1, xt2 = [x.half().cuda(non_blocking=True) for x in batch]
        xt0 = xt0.transpose(3, 2).transpose(2, 1)
        xt1 = xt1.transpose(3, 2).transpose(2, 1)
        xt2 = xt2.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(xt0)
            loss += criterion(out, xt1)
            out = model(out)
            loss += criterion(out, xt2)
        count += 1

    dist.reduce(loss, 0)
    dist.reduce(count, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def graphcast_evaluate(data_loader, graph, model, criterion, device="cuda"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)

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

    # switch to evaluation mode
    model.eval()
    for step, batch in enumerate(data_loader):
        pred_list = []
        x, y = [x.half().cuda(non_blocking=True) for x in batch]
        input_x[0] = x
        bs,ts,c,h,w = x.shape
        # y [batch, time(20), channel(70), h, w]

        for t in range(20):
            with torch.cuda.amp.autocast():
                out = model(*input_x)
                out = out.reshape(bs, h, w, c).transpose(1,3) # [bs, c, h, w]
            pred_list.append(out.unsqueeze(1))
            x = torch.concat([x[:,1:,...], out.unsqueeze(1)], dim=1)
            input_x[0] = x

        pred = torch.concat(pred_list,dim=1)
        loss = criterion(pred[:,:,-5:,...], y[:,:,-5:,...])
        loss_all += loss.item()
        count += 1

        score = run_eval(pred[...,-5:,30:-30,30:-30], y[...,-5:,30:-30,30:-30])
        score_all += score["score"]

        if step % 200 == 0:
            print("Step: ", step, " | Valid Aver Loss:", (loss_all/count).item())

    return loss_all / count


