# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from torch_geometric.nn import GATv2Conv
# from nnunet_mednext.network_architecture.mednextv1.blocks import  MedNeXtDownBlock,OutBlock,MedNeXtUpBlock,MedNeXtBlock

# class GraphCrossAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(GraphCrossAttention, self).__init__()
#         self.query_conv = GATv2Conv(in_channels=in_dim, out_channels=in_dim // 8)
#         self.key_conv = GATv2Conv(in_channels=in_dim, out_channels=in_dim // 8)
#         self.value_conv = GATv2Conv(in_channels=in_dim, out_channels=in_dim)
#         self.softmax = nn.Softmax(dim=3)
#         self.INF = lambda B, H, W: -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.merge = nn.Conv3d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=1)

#     @staticmethod
#     def create_edge_index(h, w, d):
#         edge_index = []
#         for z in range(d):
#             for y in range(h):
#                 for x in range(w):
#                     index = z * h * w + y * w + x
#                     neighbors = [
#                         (z, y, x - 1), (z, y, x + 1),
#                         (z, y - 1, x), (z, y + 1, x),
#                         (z - 1, y, x), (z + 1, y, x)
#                     ]
#                     for nz, ny, nx in neighbors:
#                         if 0 <= nz < d and 0 <= ny < h and 0 <= nx < w:
#                             neighbor_index = nz * h * w + ny * w + nx
#                             edge_index.append((index, neighbor_index))
#         return torch.tensor(edge_index, dtype=torch.long).t()

#     def forward(self, x):
#         res = x.clone().permute(0, 2, 3, 4, 1)
#         m_batchsize, height, width, depth, c = x.shape
#         edge_index = self.create_edge_index(height, width, depth).to(x.device)
#         y = x.view(m_batchsize * height * width * depth, c)
#         x = x.permute(0, 4, 1, 2, 3)

#         proj_query = self.query_conv(y, edge_index).view(m_batchsize, height, width, depth, -1)
#         proj_key = self.key_conv(y, edge_index).view(m_batchsize, height, width, depth, -1)
#         proj_value = self.value_conv(y, edge_index).view(m_batchsize, height, width, depth, -1)

#         energy = (torch.einsum("bhwdc,bhwdc->bhwdc", proj_query, proj_key)).view(m_batchsize, height, width, depth, -1)
#         attention = self.softmax(energy)
#         out = torch.einsum("bhwdc,bhwdc->bhwdc", attention, proj_value).view(m_batchsize, height, width, depth, -1)

#         y = self.gamma * out + x
#         res = torch.cat([y, res], dim=1)
#         res = self.merge(res)
#         return res




import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn import GATv2Conv
from nnunet_mednext.network_architecture.mednextv1.blocks import *
from torch.nn import Softmax

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class GraphCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(GraphCrossAttention, self).__init__()
        self.query_conv = GATv2Conv(in_channels=in_dim, out_channels=in_dim // 8)
        self.key_conv = GATv2Conv(in_channels=in_dim, out_channels=in_dim // 8)
        self.value_conv = GATv2Conv(in_channels=in_dim, out_channels=in_dim)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.merge = nn.Conv3d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=1)

    @staticmethod
    def create_edge_index(h, w, d):
        edge_index = []
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    index = z * h * w + y * w + x
                    neighbors = [
                        (z, y, x - 1), (z, y, x + 1),
                        (z, y - 1, x), (z, y + 1, x),
                        (z - 1, y, x), (z + 1, y, x)
                    ]
                    for nz, ny, nx in neighbors:
                        if 0 <= nz < d and 0 <= ny < h and 0 <= nx < w:
                            neighbor_index = nz * h * w + ny * w + nx
                            edge_index.append((index, neighbor_index))
        return torch.tensor(edge_index, dtype=torch.long).t()

    def forward(self, x):
        res = x.clone()
        m_batchsize, C, depth, height, width = x.shape
        edge_index = self.create_edge_index(height, width, depth).to(x.device)
        y = x.view(m_batchsize * depth * height * width, C)

        proj_query = self.query_conv(y, edge_index).view(m_batchsize, depth, height, width, -1)
        proj_query_H = proj_query.permute(0, 4, 2, 3, 1).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 4, 1, 3, 2).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_query_D = proj_query.permute(0, 4, 3, 2, 1).contiguous().view(m_batchsize * depth, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(y, edge_index).view(m_batchsize, depth, height, width, -1)
        proj_key_H = proj_key.permute(0, 4, 2, 3, 1).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 4, 1, 3, 2).contiguous().view(m_batchsize * height, -1, width)
        proj_key_D = proj_key.permute(0, 4, 3, 2, 1).contiguous().view(m_batchsize * depth, -1, width)

        proj_value = self.value_conv(y, edge_index).view(m_batchsize, depth, height, width, -1)
        proj_value_H = proj_value.permute(0, 4, 2, 3, 1).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 4, 1, 3, 2).contiguous().view(m_batchsize * height, -1, width)
        proj_value_D = proj_value.permute(0, 4, 3, 2, 1).contiguous().view(m_batchsize * depth, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + INF(m_batchsize, height, width).to(x.device)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        energy_D = torch.bmm(proj_query_D, proj_key_D).view(m_batchsize, depth, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        att_D = concate[:, :, :, height + width:height + width + depth].contiguous().view(m_batchsize * depth, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize, depth, -1, width).permute(0, 2, 1, 3)

        out_H = out_H.permute(0, 1, 3, 2).contiguous().view(m_batchsize, C, depth, height, width)
        out_W = out_W.permute(0, 1, 3, 2).contiguous().view(m_batchsize, C, depth, height, width)
        out_D = out_D.permute(0, 1, 3, 2).contiguous().view(m_batchsize, C, depth, height, width)

        y = self.gamma * (out_H + out_W + out_D) + x
        res = torch.cat([y, res], dim=1)
        res = self.merge(res)

        return res

# Ensure that MedNeXt is updated to use the GraphCrossAttention
class MedNextV1_graph(nn.Module):
    def __init__(self, in_channels, n_channels, n_classes, exp_r=4, kernel_size=7, deep_supervision=False, do_res=True, do_res_up_down=False, checkpoint_style=None, block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2], norm_type='group'):
        super().__init__()
        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True

        self.stem = nn.Conv3d(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(n_channels, n_channels, exp_r[0], kernel_size, do_res, norm_type)
            for i in range(block_counts[0])
        ])

        self.down_0 = MedNeXtDownBlock(n_channels, 2 * n_channels, exp_r[1], kernel_size, do_res_up_down, norm_type)

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(2 * n_channels, 2 * n_channels, exp_r[1], kernel_size, do_res, norm_type)
            for i in range(block_counts[1])
        ])

        self.down_1 = MedNeXtDownBlock(2 * n_channels, 4 * n_channels, exp_r[2], kernel_size, do_res_up_down, norm_type)

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(4 * n_channels, 4 * n_channels, exp_r[2], kernel_size, do_res, norm_type)
            for i in range(block_counts[2])
        ])

        self.down_2 = MedNeXtDownBlock(4 * n_channels, 8 * n_channels, exp_r[3], kernel_size, do_res_up_down, norm_type)

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(8 * n_channels, 8 * n_channels, exp_r[3], kernel_size, do_res, norm_type)
            for i in range(block_counts[3])
        ])

        self.down_3 = MedNeXtDownBlock(8 * n_channels, 16 * n_channels, exp_r[4], kernel_size, do_res_up_down, norm_type)

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(16 * n_channels, 16 * n_channels, exp_r[4], kernel_size, do_res, norm_type)
            for i in range(block_counts[4])
        ])

        self.up_3 = MedNeXtUpBlock(16 * n_channels, 8 * n_channels, exp_r[5], kernel_size, do_res_up_down, norm_type)
        self.graph_attn_3 = GraphCrossAttention(8 * n_channels)

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(8 * n_channels, 8 * n_channels, exp_r[5], kernel_size, do_res, norm_type)
            for i in range(block_counts[5])
        ])

        self.up_2 = MedNeXtUpBlock(8 * n_channels, 4 * n_channels, exp_r[6], kernel_size, do_res_up_down, norm_type)
        self.graph_attn_2 = GraphCrossAttention(4 * n_channels)

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(4 * n_channels, 4 * n_channels, exp_r[6], kernel_size, do_res, norm_type)
            for i in range(block_counts[6])
        ])

        self.up_1 = MedNeXtUpBlock(4 * n_channels, 2 * n_channels, exp_r[7], kernel_size, do_res_up_down, norm_type)
        self.graph_attn_1 = GraphCrossAttention(2 * n_channels)

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(2 * n_channels, 2 * n_channels, exp_r[7], kernel_size, do_res, norm_type)
            for i in range(block_counts[7])
        ])

        self.up_0 = MedNeXtUpBlock(2 * n_channels, n_channels, exp_r[8], kernel_size, do_res_up_down, norm_type)
        self.graph_attn_0 = GraphCrossAttention(n_channels)

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(n_channels, n_channels, exp_r[8], kernel_size, do_res, norm_type)
            for i in range(block_counts[8])
        ])

        self.out_0 = OutBlock(n_channels, n_classes)

        if deep_supervision:
            self.out_1 = OutBlock(2 * n_channels, n_classes)
            self.out_2 = OutBlock(4 * n_channels, n_classes)
            self.out_3 = OutBlock(8 * n_channels, n_classes)
            self.out_4 = OutBlock(16 * n_channels, n_classes)

        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            x_up_3 = self.graph_attn_3(x_up_3)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            x_up_2 = self.graph_attn_2(x_up_2)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            x_up_1 = self.graph_attn_1(x_up_1)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            x_up_0 = self.graph_attn_0(x_up_0)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            x_up_3 = self.graph_attn_3(x_up_3)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)
            if self.do_ds:
                x_ds_3 = self.out_3(x)

            x_up_2 = self.up_2(x)
            x_up_2 = self.graph_attn_2(x_up_2)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)

            x_up_1 = self.up_1(x)
            x_up_1 = self.graph_attn_1(x_up_1)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)

            x_up_0 = self.up_0(x)
            x_up_0 = self.graph_attn_0(x_up_0)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            x = self.out_0(x)

        if self.do_ds and self.training:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x

if __name__ == "__main__":
    network = MedNextV1_graph(
        in_channels=4,
        n_channels=32,
        n_classes=3,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=3,
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style=None,
    ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    x = torch.zeros((1, 1, 64, 64, 64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())

    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)


