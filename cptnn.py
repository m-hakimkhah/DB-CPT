import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.container import ModuleList
import copy
from torch.autograd import Variable
#from ptflops import get_model_complexity_info
from Backup_pesq import numParams


class DenseBlock(nn.Module): #dilated dense block
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class dense_encoder(nn.Module):
    def __init__(self, width =64):
        super(dense_encoder, self).__init__()
        self.in_channels = 2
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x


class dense_encoder_mag(nn.Module):
    def __init__(self, width =64):
        super(dense_encoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x

class Input_Module(nn.Module):
    def __init__(self, width =64):
        super(Input_Module, self).__init__()
        self.in_channels = 64
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))
        self.inp_prelu = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_conv(x))  # [b, 64, T, F]
        return out


class interction(nn.Module):
    def __init__(self, input_size, normsize):
        super(interction, self).__init__()
        self.inter = nn.Sequential(
            nn.Conv2d(2 * input_size, input_size, kernel_size=(1,1)),
            nn.LayerNorm(normsize),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        input_merge = torch.cat((input1, input2), dim =1)
        output_mask = self.inter(input_merge)
        output = input1 + input2*output_mask
        return output


class Transformer_SA(nn.Module):
    '''
    transformer with  self-attention
    '''
    def __init__(self, embed_dim, hidden_size, num_heads, bidirectional=True):
        super(Transformer_SA, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=bidirectional)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, embed_dim)
        self.dropout = nn.Dropout(0.0)
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        '''
        x: [length, batch, dimension]
        return: [length, batch, dimension]
        '''
        y = self.ln1(x)
        y, _ = self.mha(y, y, y)
        y += x
        z = self.ln2(y)
        z, _ = self.gru(z)
        z = self.gelu(z)
        z = self.fc(z)
        z = self.dropout(z)
        z += y
        z = self.ln3(z)
        return z

class Transformer_CA(nn.Module):
    '''
    transformer with  cross-attention
    '''
    def __init__(self, embed_dim, hidden_size, num_heads, bidirectional=True):
        super(Transformer_CA, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=bidirectional)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, embed_dim)
        self.dropout = nn.Dropout(0.0)
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x0, x1):
        '''
        x0, x1: [length, batch, dimension]
        return: [length, batch, dimension]
        '''
        y0 = self.ln1(x0)
        y1 = self.ln1(x1)
        y, _ = self.mha(y0, y1, y1)
        y += x0
        z = self.ln2(y)
        z, _ = self.gru(z)
        z = self.gelu(z)
        z = self.fc(z)
        z = self.dropout(z)
        z += y
        z = self.ln3(z)
        return z


class CPTB(nn.Module):
    '''
    cross-parallel transformer block (CPTB)
    '''
    def __init__(self, embed_dim, hidden_size, num_heads, num_groups):
        super(CPTB, self).__init__()
        self.frequency_transformer = Transformer_SA(embed_dim, hidden_size, num_heads)
        self.frequency_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

        self.time_transformer = Transformer_SA(embed_dim, hidden_size, num_heads)
        self.time_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

        self.fusion_transformer = Transformer_CA(embed_dim, hidden_size, num_heads)
        self.fusion_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

    def forward(self, x):

        B, C, T, F = x.shape

        Frequency_feat = x.permute(3, 0, 2, 1).contiguous().view(F, B*T, -1)  # [F, B*T, C]
        Time_feat = x.permute(2, 0, 3, 1).contiguous().view(T, B*F, -1) # [T, B*F, C]

        Frequency_feat = self.frequency_transformer(Frequency_feat)     # F BT C
        Frequency_feat = Frequency_feat.view(F, B, T, -1).permute(1, 3, 2, 0).contiguous()  # [B C T F]
        Frequency_feat = self.frequency_norm(Frequency_feat)
        Frequency_feat = Frequency_feat.permute(3, 0, 2, 1).contiguous().view(F, B*T, -1)  # [F, B*T, C]

        Time_feat = self.time_transformer(Time_feat)                    # T BF C
        Time_feat = Time_feat.view(T, B, F, -1).permute(1, 3, 0, 2).contiguous()  # [B C T F]
        Time_feat = self.time_norm(Time_feat)
        Time_feat = Time_feat.permute(3, 0, 2, 1).contiguous().view(F, B*T, -1) # [F, B*T, C]

        fusion_feat = self.fusion_transformer(Frequency_feat, Time_feat)  # F BT C
        fusion_feat = fusion_feat.view(F, B, T, -1).permute(1, 3, 2, 0).contiguous()  # [B C T F]
        fusion_feat = self.fusion_norm(fusion_feat)

        #fusion_feat = fusion_feat.view(T, B, F, -1).permute(1, 3, 0, 2).contiguous()  # B C T F

        #fusion_feat = torch.reshape(fusion_feat, [B, C, T, F])

        return fusion_feat

class CPTM(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_groups, cptm_layers):
        super(CPTM, self).__init__()
        self.layers = cptm_layers
        self.net = nn.ModuleList()
        for i in range(cptm_layers):
            self.net.append(CPTB(embed_dim, hidden_size, num_heads, num_groups))

    def forward(self, x):
        '''
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        '''
        
        for i in range(self.layers):
            y = self.net[i](x)
            x = x + y
        return x


class SPConvTranspose2d(nn.Module): #sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class dense_decoder(nn.Module):
    def __init__(self, width =64):
        super(dense_decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width =width
        self.dec_dense1 = DenseBlock(161, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.dec_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1))
        self.dec_norm2 = nn.LayerNorm(161)
        self.dec_prelu2 = nn.PReLU(self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(x)))))
        out = self.dec_dense1(out)
        out = self.dec_prelu2(self.dec_norm2(self.dec_conv2(out)))

        out = self.out_conv(out)
        return out



class dense_decoder_masking(nn.Module):
    def __init__(self, width =64):
        super(dense_decoder_masking, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width =width
        self.dec_dense1 = DenseBlock(161, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.dec_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1))
        self.dec_norm2 = nn.LayerNorm(161)
        self.dec_prelu2 = nn.PReLU(self.width)
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))
        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1,1)),
            nn.Sigmoid()
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1,1)),
            nn.Tanh()
        )
        self.mask_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(1, 1))
        self.maskPRelu = nn.PReLU(161, init=-0.25)

    def forward(self, x):
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(x)))))
        out = self.dec_dense1(out)
        out = self.dec_prelu2(self.dec_norm2(self.dec_conv2(out)))
        out = self.out_conv(out)
        out = self.mask1(out) * self.mask2(out)
        out = self.mask_conv(out).permute(0, 3, 2, 1).squeeze(-1)
        out = self.maskPRelu(out).permute(0, 2, 1).unsqueeze(1)
        return out



class DBCPTNN(nn.Module):
    def __init__(self,
                 feat_dim=64,
                 hidden_size=64,
                 num_heads=4,
                 num_groups=4,
                 cptm_layers=2):
        super(DBCPTNN, self).__init__()

        self.encoder_M = dense_encoder_mag()
        self.encoder_C = dense_encoder()

        self.inter_m1 = interction(64, 80)
        self.inter_c1 = interction(64, 80)
        self.inter_m2 = interction(64, 80)
        self.inter_c2 = interction(64, 80)

        self.input_m1 = Input_Module()
        self.input_c1 = Input_Module()

        self.cptm_M1 = CPTM(feat_dim, hidden_size, num_heads, num_groups, cptm_layers)
        self.cptm_C1 = CPTM(feat_dim, hidden_size, num_heads, num_groups, cptm_layers)

        self.cptm_M2 = CPTM(feat_dim, hidden_size, num_heads, num_groups, cptm_layers)
        self.cptm_C2 = CPTM(feat_dim, hidden_size, num_heads, num_groups, cptm_layers)

        self.mask_decoder_mag = dense_decoder_masking()

        self.decoder_R = dense_decoder()
        self.decoder_I = dense_decoder()

    def forward(self, x):

        x_mag_ri, x_phase = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        x_mag = x_mag_ri.unsqueeze(dim = 1)

        x_mag1 = self.encoder_M(x_mag)
        x_com1 = self.encoder_C(x)

        x_mag2 = self.inter_m1(x_mag1, x_com1)
        x_com2 = self.inter_c1(x_com1, x_mag1)

        x_mag2 = self.input_m1(x_mag2)
        x_com2 = self.input_c1(x_com2)

        y_mag1 = self.cptm_M1(x_mag2)
        y_com1 = self.cptm_C1(x_com2)

        y_mag = self.inter_m2(y_mag1, y_com1)
        y_com = self.inter_c2(y_com1, y_mag1)

        y_mag = self.cptm_M2(y_mag)
        y_com = self.cptm_C2(y_com)
        
        # magnitude decode
        y_mag = self.mask_decoder_mag(y_mag)
        y_mag = y_mag.squeeze(dim=1)

        # real and imag decode
        x_real = self.decoder_R(y_com)
        x_imag = self.decoder_I(y_com)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim = 1)

        # magnitude and ri components        
        y_mag = y_mag * x_mag_ri
        x_r_out,x_i_out = (y_mag * torch.cos(x_phase) + x_real), (y_mag * torch.sin(x_phase)+ x_imag)

        x_com_out = torch.stack((x_r_out,x_i_out),dim=1)


        return x_com_out
