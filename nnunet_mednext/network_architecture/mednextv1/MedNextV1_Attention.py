from nnunet_mednext.network_architecture.mednextv1.blocks import *
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class EnhancedAttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_heads=4):
        super(EnhancedAttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv3d(F_g, F_g, kernel_size=1, stride=1, padding=0, bias=True)
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=F_int, num_heads=num_heads)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        out = out + self.residual(x)  # Add residual connection
        return out

class MedNeXt_attention(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 n_channels: int,
                 n_classes: int, 
                 exp_r: int = 4, 
                 kernel_size: int = 7, 
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False, 
                 do_res: bool = True, 
                 do_res_up_down: bool = False, 
                 checkpoint_style: bool = None, 
                 block_counts: list = [2,2,2,2,2,2,2,2,2], 
                 norm_type = 'group'):
        super().__init__()
        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        self.stem = nn.Conv3d(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                ) 
            for i in range(block_counts[0])]
        ) 

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )
    
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                )
            for i in range(block_counts[1])]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                )
            for i in range(block_counts[2])]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )
        
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                )            
            for i in range(block_counts[3])]
        )
        
        self.down_3 = MedNeXtDownBlock(
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                )
            for i in range(block_counts[4])]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )
        self.Att3 = EnhancedAttentionBlock3D(F_g=8*n_channels, F_l=8*n_channels, F_int=4*n_channels)

        self.reduce_channels_3 = nn.Conv3d(16*n_channels, 8*n_channels, kernel_size=1)

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=8*n_channels,
                out_channels=n_channels*8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                )
            for i in range(block_counts[5])]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )
        self.Att2 = EnhancedAttentionBlock3D(F_g=4*n_channels, F_l=4*n_channels, F_int=2*n_channels)

        self.reduce_channels_2 = nn.Conv3d(8*n_channels, 4*n_channels, kernel_size=1)

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=4*n_channels,
                out_channels=n_channels*4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )
        self.Att1 = EnhancedAttentionBlock3D(F_g=2*n_channels, F_l=2*n_channels, F_int=n_channels)

        self.reduce_channels_1 = nn.Conv3d(4*n_channels, 2*n_channels, kernel_size=1)

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=2*n_channels,
                out_channels=n_channels*2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type
                )
            for i in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type
        )
        self.Att0 = EnhancedAttentionBlock3D(F_g=n_channels, F_l=n_channels, F_int=n_channels//2)

        self.reduce_channels_0 = nn.Conv3d(2*n_channels, n_channels, kernel_size=1)

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type
                )
            for i in range(block_counts[8])]
        )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes)

        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes)
            self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes)

        self.block_counts = block_counts

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
            x_up_3 = self.Att3(g=x_up_3, x=x_res_3)  # Apply attention here
            dec_x = torch.cat((x_up_3, x_res_3), dim=1) 
            dec_x = self.reduce_channels_3(dec_x)
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            x_up_2 = self.Att2(g=x_up_2, x=x_res_2)  # Apply attention here
            dec_x = torch.cat((x_up_2, x_res_2), dim=1)
            dec_x = self.reduce_channels_2(dec_x)
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            x_up_1 = self.Att1(g=x_up_1, x=x_res_1)  # Apply attention here
            dec_x = torch.cat((x_up_1, x_res_1), dim=1)
            dec_x = self.reduce_channels_1(dec_x)
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            x_up_0 = self.Att0(g=x_up_0, x=x_res_0)  # Apply attention here
            dec_x = torch.cat((x_up_0, x_res_0), dim=1)
            dec_x = self.reduce_channels_0(dec_x)
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

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
            x_up_3 = self.Att3(g=x_up_3, x=x_res_3)  # Apply attention here
            dec_x = torch.cat((x_up_3, x_res_3), dim=1) 
            dec_x = self.reduce_channels_3(dec_x)
            x = self.dec_block_3(dec_x)
            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            x_up_2 = self.Att2(g=x_up_2, x=x_res_2)  # Apply attention here
            dec_x = torch.cat((x_up_2, x_res_2), dim=1)
            dec_x = self.reduce_channels_2(dec_x)
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            x_up_1 = self.Att1(g=x_up_1, x=x_res_1)  # Apply attention here
            dec_x = torch.cat((x_up_1, x_res_1), dim=1)
            dec_x = self.reduce_channels_1(dec_x)
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            x_up_0 = self.Att0(g=x_up_0, x=x_res_0)  # Apply attention here
            dec_x = torch.cat((x_up_0, x_res_0), dim=1)
            dec_x = self.reduce_channels_0(dec_x)
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if (self.do_ds==True) and (self.training==True):
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            return x

if __name__ == "__main__":

    network = MedNeXt_attention(
            in_channels = 1, 
            n_channels = 32,
            n_classes = 13,
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = None,
            
        ).cuda()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    x = torch.zeros((1,1,64,64,64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())
    
    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)

