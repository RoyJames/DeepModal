import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN , Dropout
from torch_geometric.nn import DynamicEdgeConv, global_max_pool


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Feature_extractor(torch.nn.Module):
    def __init__(self, k=30, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        # self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
        #                Dropout(0.5), Lin(128, out_channels))

    def forward(self, data):
        pos, normal, batch = data.pos, data.normal, data.batch
        x0 = torch.cat([pos,normal], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        return out
        # out = self.mlp(out)
        # return F.log_softmax(out, dim=1)

class dgcnn_segmentation(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.extractor = Feature_extractor()
        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                        Dropout(0.5), Lin(128, out_channels))

    def forward(self, data):
        mid_feature = self.extractor(data)
        out = self.mlp(mid_feature)
        return out

class dgcnn_classification(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.extractor = Feature_extractor()
        self.mlp = Seq(MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        mid_feature = global_max_pool(self.extractor(data), data.batch)
        out = self.mlp(mid_feature)
        return out