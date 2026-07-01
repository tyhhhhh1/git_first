import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + 1e-8)
        return x_norm * self.gamma + self.beta

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                        stride=1, padding=kernel_size // 2, groups=d_model)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_conv = self.layernorm(x).transpose(1, 2)
        x_conv = self.pointwise_conv1(x_conv)
        x_conv = F.glu(x_conv, dim=1)
        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.batchnorm(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.pointwise_conv2(x_conv)
        x_conv = self.dropout(x_conv)
        return x_conv.transpose(1, 2)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, conv_kernel_size=15):
        super().__init__()
        self.pe = PositionalEncoding1D(d_model, max_len=1024)
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        x = self.pe(x)
        x2 = self.norm1(x)
        attn_out, _ = self.mha(x2, x2, x2)
        x = x + attn_out
        x = x + self.conv_module(x)
        x = x + self.ffn(self.norm2(x))
        return x

class DualPathConformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.intra_conformer = ConformerBlock(d_model, nhead, conv_kernel_size=15)
        self.inter_conformer = ConformerBlock(d_model, nhead, conv_kernel_size=31)

    def forward(self, x):
        B, D, K, S = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * S, K, D)
        x = self.intra_conformer(x)
        x = x.reshape(B, S, K, D).permute(0, 3, 2, 1)

        x = x.permute(0, 2, 3, 1).reshape(B * K, S, D)
        x = self.inter_conformer(x)
        x = x.reshape(B, K, S, D).permute(0, 3, 1, 2)
        return x

class NMRSepFormer(nn.Module):
    def __init__(self, num_compounds=20, d_model=128, nhead=8, num_layers=3,
                 chunk_size=128, hop_size=64):
        super().__init__()
        self.num_compounds = num_compounds
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.hop_size = hop_size

        self.encoder_conv = nn.Conv1d(2, d_model * 2, kernel_size=16, stride=4, padding=6)
        self.norm = GlobalLayerNorm(d_model)

        self.layers = nn.ModuleList([
            DualPathConformerBlock(d_model, nhead) for _ in range(num_layers)
        ])

        self.pre_mask_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.mask_conv = nn.Conv1d(d_model, d_model * num_compounds, kernel_size=1)

        self.prototypes = nn.Parameter(torch.randn(num_compounds, d_model) * 0.02)
        self.gate_scale = nn.Parameter(torch.ones(num_compounds) * 1.5)
        self.gate_threshold = nn.Parameter(torch.ones(num_compounds) * (-0.2))

        self.shared_decoder = nn.ConvTranspose1d(d_model, d_model, kernel_size=16, stride=4, padding=6)
        self.output_heads = nn.ModuleList([
            nn.Conv1d(d_model, 1, kernel_size=1) for _ in range(num_compounds)
        ])
        
        # ========================================================
        # === 新增：物理重构面积约束的核心——背景基线解码器 ===
        # ========================================================
        self.bg_decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, 64, kernel_size=16, stride=4, padding=6),
            nn.SiLU(),
            nn.Conv1d(64, 1, kernel_size=31, padding=15), 
            nn.ReLU() 
        )

    def forward(self, x):
        x = self.encoder_conv(x)           
        x = F.glu(x, dim=1)                
        orig_enc = x
        x = self.norm(x)
        B, D, T = x.shape

        x_padded = x.unsqueeze(3)
        chunks = F.unfold(x_padded, kernel_size=(self.chunk_size, 1),
                          stride=(self.hop_size, 1))
        S = chunks.size(-1)                
        chunks = chunks.view(B, D, self.chunk_size, S)

        for layer in self.layers:
            chunks = layer(chunks)

        window = torch.hann_window(self.chunk_size, periodic=True, device=chunks.device)
        window = window.view(1, 1, self.chunk_size, 1)          
        chunks = chunks * window                                

        chunks_flat = chunks.reshape(B, D * self.chunk_size, S)
        out_len = (S - 1) * self.hop_size + self.chunk_size
        x_recon = F.fold(chunks_flat, output_size=(out_len, 1),
                         kernel_size=(self.chunk_size, 1),
                         stride=(self.hop_size, 1))
        x_recon = x_recon.squeeze(3)

        win_flat = window.expand(B, D, self.chunk_size, S).reshape(B, D * self.chunk_size, S)
        norm_factor = F.fold(win_flat, output_size=(out_len, 1),
                             kernel_size=(self.chunk_size, 1),
                             stride=(self.hop_size, 1))
        norm_factor = norm_factor.squeeze(3)
        x_recon = x_recon / (norm_factor + 1e-8)

        if x_recon.size(2) > T:
            x_recon = x_recon[:, :, :T]
        elif x_recon.size(2) < T:
            x_recon = F.pad(x_recon, (0, T - x_recon.size(2)))

        x_dec = self.pre_mask_conv(x_recon)    
        masks = self.mask_conv(x_dec)           
        masks = torch.split(masks, self.d_model, dim=1)  

        separated_outputs = []
        for i in range(self.num_compounds):
            m = torch.sigmoid(masks[i])                 
            shared = self.shared_decoder(orig_enc * m)  
            out = F.leaky_relu(self.output_heads[i](shared), negative_slope=0.1)

            proto_norm = F.normalize(self.prototypes[i:i+1], dim=-1)       
            feat_norm = F.normalize(x_dec, dim=1)                        
            sim = torch.einsum('bd,bdt->bt', proto_norm.expand(B, -1), feat_norm)  
            gate = torch.sigmoid(self.gate_scale[i] * (sim - self.gate_threshold[i]))  

            gate_up = F.interpolate(gate.unsqueeze(1), size=16384, mode='nearest')                           
            out = out * gate_up
            separated_outputs.append(out)

        output = torch.stack(separated_outputs, dim=1).squeeze(2)  

        # ========================================================
        # === 新增：解耦输出大分子背景基线 ===
        # ========================================================
        background = self.bg_decoder(x_dec) 
        background = F.avg_pool1d(background, kernel_size=63, stride=1, padding=31)

        return output, background

    @torch.no_grad()
    def init_prototypes_from_templates(self, templates_dict, device):
        for idx, name in enumerate(templates_dict):
            spec = templates_dict[name]                               
            temp_lin = torch.from_numpy(spec).float().to(device)
            temp_log = torch.log10(torch.clamp(temp_lin, 0.0, None) + 1e-6)
            dual = torch.stack([temp_lin, temp_log], dim=0).unsqueeze(0)  
            enc = self.encoder_conv(dual)                             
            enc = F.glu(enc, dim=1)                                   
            enc = self.norm(enc)
            self.prototypes.data[idx] = enc.mean(dim=-1).squeeze(0)