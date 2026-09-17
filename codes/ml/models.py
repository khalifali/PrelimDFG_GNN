"""Residual GINE architecture verified against the newly uploaded v9 source."""
import numpy as np
import torch
from torch import nn


def batch_graphs(records, node_mean, node_scale, rotate=False):
    nodes, pos, edges, assignments = [], [], [], []
    offset = 0
    for i, g in enumerate(records):
        p = torch.as_tensor(g['pos'], dtype=torch.float32)
        if rotate:
            rotation, _ = torch.linalg.qr(torch.randn(3, 3))
            rotation[:, 0] *= torch.linalg.det(rotation)
            p = p @ rotation
        nodes.append(torch.as_tensor((g['nodes'] - node_mean) / node_scale, dtype=torch.float32))
        pos.append(p)
        edges.append(torch.as_tensor(g['edge'], dtype=torch.long) + offset)
        assignments.append(torch.full((len(p),), i, dtype=torch.long))
        offset += len(p)
    return torch.cat(nodes), torch.cat(pos), torch.cat(edges, dim=1), torch.cat(assignments)


class GNN(nn.Module):
    """Same forward architecture and state-dict layout as uploaded v9 Model."""
    def __init__(self, q=1, width=64, layers=6):
        super().__init__()
        from torch_geometric.nn import GINEConv
        self.node = nn.Linear(5, width)
        self.edge = nn.Linear(4, width)
        self.layers = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width)))
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.dropout = nn.Dropout(.1)
        self.to_latent = nn.Linear(3*width, q)
        self.head = nn.Sequential(nn.Linear(q,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self, batch, return_latent=False):
        from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
        nodes, pos, edges, assignment = batch
        src, dst = edges
        relative = pos[dst] - pos[src]
        x = self.node(torch.cat([nodes,pos],dim=1))
        edge = self.edge(torch.cat([relative.norm(dim=1,keepdim=True),relative],dim=1))
        for layer,norm in zip(self.layers,self.norms):
            update = self.dropout(torch.relu(layer(x,edges,edge)))
            x = norm(x+update)
        count = global_add_pool(torch.ones_like(x[:,:1]),assignment)
        pooled = torch.cat([global_mean_pool(x,assignment),global_max_pool(x,assignment),
                            global_add_pool(x,assignment)/torch.sqrt(count.clamp_min(1.))],dim=1)
        q = self.to_latent(pooled)
        prediction = self.head(q).flatten()
        return (prediction,q) if return_latent else prediction


class ANN(nn.Module):
    def __init__(self, q=0):
        super().__init__()
        if q:
            self.latent = nn.Sequential(nn.Linear(6,64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64,q))
            self.head = nn.Sequential(nn.Linear(q,16), nn.ReLU(), nn.Linear(16,1))
        else:
            self.latent = nn.Sequential(nn.Linear(6,64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64,32), nn.ReLU())
            self.head = nn.Linear(32,1)

    def forward(self, x, return_latent=False):
        q = self.latent(x)
        pred = self.head(q).flatten()
        return (pred, q) if return_latent else pred


def make_model(kind, q):
    return GNN(q) if kind == 'gnn' else ANN(q)


def scaler(records):
    node = np.concatenate([g['nodes'] for g in records])
    desc = np.stack([g['descriptors'] for g in records])
    y = np.array([g['target'] for g in records])
    def parameters(values):
        scale = values.std(axis=0)
        return values.mean(axis=0), np.where(scale > 1e-8, scale, 1.)
    nm, ns = np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)  # v9 uses raw node scalars
    dm, ds = parameters(desc)
    ym, ys = float(y.mean()), max(float(y.std(ddof=1)), 1e-8)
    return dict(node_mean=nm, node_scale=ns, descriptor_mean=dm, descriptor_scale=ds,
                target_mean=float(ym), target_scale=float(ys))


def inputs(records, scale, kind, rotate=False):
    if kind == 'gnn':
        return batch_graphs(records, scale['node_mean'], scale['node_scale'], rotate)
    return torch.as_tensor((np.stack([g['descriptors'] for g in records])-scale['descriptor_mean'])/scale['descriptor_scale'], dtype=torch.float32)


def predict(model, records, scale, kind, batch_size=16):
    model.eval()
    values, latent = [], []
    with torch.no_grad():
        for start in range(0,len(records),batch_size):
            p, q = model(inputs(records[start:start+batch_size],scale,kind), return_latent=True)
            values.extend((p.numpy().astype(np.float64)*float(scale['target_scale'])+float(scale['target_mean'])).tolist())
            latent.extend(q.numpy().tolist())
    return np.array(values), np.array(latent)
