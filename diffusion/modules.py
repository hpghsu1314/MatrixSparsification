#Credits to Dominic Rampas for the tutorial (aka dome272 on GitHub)

import torch
import torch.nn as nn
import torch.nn.functional as functional

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Attention(nn.Module):
    
    def __init__(self, channels, size):
        super(Attention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.LayerNorm([channels])
        self.feed_forward = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    
    def forward(self, input):
        input = input.view(-1, self.channels, self.size * self.size).swapaxes(1,2)
        input_norm = self.norm(input)
        attention_value, __ = self.mha(input_norm, input_norm, input_norm)
        attention_value = attention_value + input
        attention_value = self.feed_forward(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)
    
    
class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )
        
    def forward(self, input):
        if self.residual:
            return functional.gelu(self.double_conv(input) + input)
        else:
            return self.double_conv(input)
        
        
class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels, embedding_dim=256):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )
        
        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels)
        )
        
    def forward(self, input, timestep):
        input = self.maxpool(input)
        embedding = self.embedding_layer(timestep)[:, :, None, None].repeat(1, 1, input.shape[-2], input.shape[-1]) #may need to change this
        return input + embedding
    
    
class Upward(nn.Module):
    
    def __init__(self, in_channels, out_channels, embedding_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upward = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )
        
        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels)
        )
        
    def forward(self, input, skip_inputs, timestep):
        input = self.upward(input)
        input = torch.cat([skip_inputs, input], dim=1)
        input = self.conv(input)
        embedding = self.embedding_layer(timestep)[:, :, None, None].repeat(1, 1, input.shape[-2], input.shape[-1])
        return input + embedding
    
class conditional_UNET(nn.Module):
    
    def __init__(self, in_channel=3, out_channel=3, time_dim=256, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.input_layer = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.sa1 = Attention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = Attention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = Attention(256, 8)
        
        self.bottleneck1 = DoubleConv(256, 512)
        self.bottleneck2 = DoubleConv(512, 512)
        self.bottleneck3 = DoubleConv(512, 256)
        
        self.up1 = Upward(512, 128)
        self.sa4 = Attention(128, 16)
        self.up2 = Upward(256, 64)
        self.sa5 = Attention(64, 32)
        self.up3 = Upward(128, 64)
        self.sa6 = Attention(64, 64)
        self.output_layer = nn.Conv2d(64, out_channel, kernel_size=1)
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        
    def pos_encoding(self, timestep, channels):
        inv_freq = 1.0/10000 ** (torch.arange(0, channels, 2).float()/channels)
        pos_enc_a = torch.sin(timestep.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(timestep.repeat(1, channels//2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        if y is not None:
            t += self.label_emb(y)
            
        x1 = self.input_layer(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bottleneck1(x4)
        x4 = self.bottleneck2(x4)
        x4 = self.bottleneck3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.output_layer(x)
        return output
    
if __name__ == "__main__":
    net = conditional_UNET(num_classes=10)
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)