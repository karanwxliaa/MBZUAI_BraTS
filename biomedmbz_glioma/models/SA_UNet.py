import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock3D(nn.Module):
    def __init__(self, block_size, keep_prob):
        super(DropBlock3D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        gamma = self._compute_gamma(x)
        mask = (torch.rand_like(x) < gamma).float()
        mask = F.max_pool3d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        return x * mask * mask.numel() / mask.sum()

    def _compute_gamma(self, x):
        return ((1.0 - self.keep_prob) / (self.block_size ** 3)) * \
               (x.size(2) * x.size(3) * x.size(4) / ((x.size(2) - self.block_size + 1) * 
                                                    (x.size(3) - self.block_size + 1) * 
                                                    (x.size(4) - self.block_size + 1)))

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x * self.sigmoid(x)

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, block_size, keep_prob):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropblock1 = DropBlock3D(block_size, keep_prob)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropblock2 = DropBlock3D(block_size, keep_prob)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropblock1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropblock2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class SAUNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, start_neurons=16, block_size=7, keep_prob=0.9):
        super(SAUNet3D, self).__init__()
        self.start_neurons = start_neurons
        self.block_size = block_size
        self.keep_prob = keep_prob

        self.conv1 = ConvBlock3D(in_channels, start_neurons * 1, block_size, keep_prob)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = ConvBlock3D(start_neurons * 1, start_neurons * 2, block_size, keep_prob)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = ConvBlock3D(start_neurons * 2, start_neurons * 4, block_size, keep_prob)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = ConvBlock3D(start_neurons * 4, start_neurons * 8, block_size, keep_prob)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.conv_middle = ConvBlock3D(start_neurons * 8, start_neurons * 16, block_size, keep_prob)
        self.spatial_attention = SpatialAttention3D()

        self.up4 = nn.ConvTranspose3d(start_neurons * 16, start_neurons * 8, kernel_size=2, stride=2)
        self.conv_up4 = ConvBlock3D(start_neurons * 16, start_neurons * 8, block_size, keep_prob)

        self.up3 = nn.ConvTranspose3d(start_neurons * 8, start_neurons * 4, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock3D(start_neurons * 8, start_neurons * 4, block_size, keep_prob)

        self.up2 = nn.ConvTranspose3d(start_neurons * 4, start_neurons * 2, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock3D(start_neurons * 4, start_neurons * 2, block_size, keep_prob)

        self.up1 = nn.ConvTranspose3d(start_neurons * 2, start_neurons * 1, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock3D(start_neurons * 2, start_neurons * 1, block_size, keep_prob)

        self.conv_final = nn.Conv3d(start_neurons * 1, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv_middle = self.conv_middle(pool4)
        conv_middle = self.spatial_attention(conv_middle)

        up4 = self.up4(conv_middle)
        up4 = torch.cat([up4, conv4], dim=1)
        up4 = self.conv_up4(up4)

        up3 = self.up3(up4)
        up3 = torch.cat([up3, conv3], dim=1)
        up3 = self.conv_up3(up3)

        up2 = self.up2(up3)
        up2 = torch.cat([up2, conv2], dim=1)
        up2 = self.conv_up2(up2)

        up1 = self.up1(up2)
        up1 = torch.cat([up1, conv1], dim=1)
        up1 = self.conv_up1(up1)

        out = self.conv_final(up1)
        return self.softmax(out)

