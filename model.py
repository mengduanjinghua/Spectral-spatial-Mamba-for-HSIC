import torch
import torch.nn as nn
from mamba import SSM


from functools import partial
from typing import Callable
from timm.models.layers import DropPath

from pos_embed import get_2d_sincos_pos_embed



class block_1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        bi: bool = True,
        cls: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.bi = bi
        self.cls = cls

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x1 = self.self_attention(x)
        if self.bi:
            if self.cls:
                x2 = x[:,0:-1,:]
                cls_token = x[:,-1:,:]
                x2 = torch.flip(x2, dims=[1])
                x2 = torch.cat((x2, cls_token), dim=1)
                x3 = self.self_attention(x2)

                x2 = x3[:,0:-1,:]
                cls_token = x3[:,-1:,:]
                x3 = torch.flip(x2, dims=[1])
                x3 = torch.cat((x3, cls_token), dim=1)
            else:
                x3 = torch.flip(x, dims=[1])
                x3 = self.self_attention(x3)
                x3 = torch.flip(x3, dims=[1])
            return self.drop_path((x1+x3)/2) + input
        else:
            return self.drop_path(x1) + input

def positional_embedding_1d(seq_len, embed_size):
    position_enc = torch.zeros(seq_len, embed_size)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
    position_enc[:, 0::2] = torch.sin(position.float() * div_term)
    position_enc[:, 1::2] = torch.cos(position.float() * div_term)
    return position_enc.unsqueeze(0)

class spectral_spatial_block(nn.Module):
    def __init__(self, embed_dim, bi=False, N=8, drop_path=0.0, norm_layer=nn.LayerNorm, cls = True, fu = True):
        super(spectral_spatial_block, self).__init__()
        self.spa_block = block_1D(
            # This module uses roughly 3 * expand * d_model^2 parameters
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls,
            # gaussian = True
            )
        self.spe_block = block_1D(
            # This module uses roughly 3 * expand * d_model^2 parameters
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls
            )
        self.linear = nn.Linear(N, N)
        self.norm = norm_layer(embed_dim)
        self.l1= nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias = False),
            nn.Sigmoid(),)
        self.fu = fu
    def forward(self, x_spa, x_spe):
        ###  x:(B, L, D)
        x_spa = self.spa_block(x_spa)   #(N, HW/P^2, D)
        B, N, D = x_spa.shape
        x_spe = self.spe_block(x_spe)   #(N, B, D)
        _,N1,_ = x_spe.shape

        if self.fu:
            x_spa_c = x_spa[:,(N-1)//2,:]
            x_spe_c = x_spe.mean(1)
            sig = self.l1((x_spa_c+x_spe_c)/2).unsqueeze(1)
            x_spa = x_spa*sig.expand(-1,N,-1)
            x_spe = x_spe*sig.expand(-1,N1,-1)

        return x_spa, x_spe



class PatchEmbed_1D(nn.Module):
    """ 1D signal to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=2, in_chans=1, embed_dim=64, norm_layer=None):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)
        x = x.transpose(1, 2)  # BCL -> BLC
        x = self.norm(x)
        return x

class PatchEmbed_2D(nn.Module):
    """ 2D image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, embed_dim=64, norm_layer=None, flatten = True):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            # x = spiral_flatten(x).transpose(1, 2)
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # x = s_flatten(x).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchEmbed_Spe(nn.Module):
    """ 2D image to Patch Embedding
    """
    def __init__(self, img_size=(9,9), patch_size=2, embed_dim=64, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv1d(
            in_channels=img_size[0]*img_size[1],
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        x = x.transpose(2,1)
        x = self.proj(x)
        x = x.transpose(2,1)
        x = self.norm(x)
        return x

class mamba_1D_model(nn.Module):
    """
    从（27 27）中中心裁取（9 9）的patch，然后通过1D卷积提取特征
    """
    def __init__(self, img_size=(3,3), spa_img_size=(224, 224), nband=107, patch_size=1, embed_dim=128, nclass=10, drop_path=0.0, depth=4, bi=True, norm_layer=nn.LayerNorm, global_pool=True, cls = True):
        super().__init__()
        self.patch_embed = PatchEmbed_Spe(img_size, patch_size, embed_dim, norm_layer)
        if nband % patch_size == 0:
          self.num_patch = nband // patch_size
        else:
          self.num_patch = nband // patch_size 
        self.half_spe_patch_size = img_size[0] // 2
        self.half_spa_patch_size = spa_img_size[0] // 2
        self.cls = cls
        if self.cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cs =  -1
        else:
            self.cs = self.num_patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
                block_1D(hidden_dim=embed_dim, drop_path = drop_path, bi = bi, cls = cls) for i in range(depth)
            ])
        self.head = nn.Linear(embed_dim, nclass)
        self.pos_embed = nn.Parameter(positional_embedding_1d(self.num_patch+1, embed_dim), requires_grad = False)
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        x_spe = x[:,:,self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1,
                            self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1]
        x = self.patch_embed(x_spe)
        # append cls token
        x = x + self.pos_embed[:, :-1, :]
        if self.cls:
            cls_token = self.cls_token + self.pos_embed[:, -1:, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)      
            x = torch.cat((x, cls_tokens), dim=1)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 0:self.cs, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, -1]
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class mamba_2D_model(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, hid_chans = 32, embed_dim=128, nclass=10, drop_path=0.0, depth=4, bi=True, norm_layer=nn.LayerNorm, global_pool=True, cls = True):
        super().__init__()

        self.dimen_redu = nn.Sequential(
                  nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
                  nn.BatchNorm2d(hid_chans),
                  nn.ReLU(),

                  nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
                  nn.BatchNorm2d(hid_chans),
                #   nn.ReLU(),
            )

        self.patch_embed = PatchEmbed_2D(img_size, patch_size, hid_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls = cls
        if self.cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cs =  -1
        else:
            self.cs = num_patches
        self.blocks = nn.ModuleList([
                  block_1D(hidden_dim=embed_dim, drop_path = drop_path,bi = bi, cls = cls) for i in range(depth)
                          ])
        self.head = nn.Linear(embed_dim, nclass)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
        self.initialize_weights()
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward_features(self, x):
        x = self.dimen_redu(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :-1, :]
        # append cls token
        if self.cls:
            cls_token = self.cls_token + self.pos_embed[:, -1:, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 0:self.cs, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, -1]
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class mamba_SS_model(nn.Module):
    def __init__(self, spa_img_size=(224, 224),spe_img_size=(5,5), spa_patch_size=16, spe_patch_size=2, in_chans=3, hid_chans = 32, embed_dim=128, nclass=10, drop_path=0.0, depth=4, bi=True, 
                 norm_layer=nn.LayerNorm, global_pool=False, cls = True, fu=True):
        super().__init__()

        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            # nn.GroupNorm(4, hid_chans),
            nn.ReLU(),

            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            # nn.GroupNorm(4, hid_chans),
            # nn.SiLU(),
            )

        self.half_spa_patch_size = spa_img_size[0] // 2
        self.half_spe_patch_size = spe_img_size[0] // 2
        self.spe_patch_embed = PatchEmbed_Spe(img_size=spe_img_size, patch_size=spe_patch_size, embed_dim=embed_dim)
        self.spa_patch_embed = PatchEmbed_2D(spa_img_size, spa_patch_size, hid_chans, embed_dim)
        spa_num_patches = self.spa_patch_embed.num_patches
        if in_chans % spe_patch_size ==0:
          spe_num_patches = in_chans//spe_patch_size
        else:
          spe_num_patches = in_chans//spe_patch_size

        self.cls = cls
        if self.cls:
          self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          N = spa_num_patches+spe_num_patches+2
          self.cs = -1
        else:
          N = spa_num_patches+spe_num_patches
          self.cs = N

        self.blocks = nn.ModuleList([
                spectral_spatial_block(embed_dim, bi, N=N, drop_path = drop_path, cls = self.cls, fu = fu) for i in range(depth)
                        ])
        self.head = nn.Linear(embed_dim, nclass)
        self.spa_pos_embed = nn.Parameter(torch.zeros(1, spa_num_patches+1, embed_dim), requires_grad=False)
        self.spe_pos_embed = nn.Parameter(positional_embedding_1d(spe_num_patches+1, embed_dim), requires_grad=False)

        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
        self.initialize_weights()
        


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        spa_pos_embed = get_2d_sincos_pos_embed(self.spa_pos_embed.shape[-1], int(self.spa_patch_embed.num_patches**.5), cls_token=True)
        self.spa_pos_embed.data.copy_(torch.from_numpy(spa_pos_embed).float().unsqueeze(0))


    def forward_features(self, x):
        x_spa = self.dimen_redu(x)
        x_spa = self.spa_patch_embed(x_spa)
        x_spa = x_spa + self.spa_pos_embed[:, :-1, :]
        # append cls token
        if self.cls:
          spa_cls_token = self.spa_cls_token + self.spa_pos_embed[:, -1:, :]
          spa_cls_tokens = spa_cls_token.expand(x_spa.shape[0], -1, -1)
          x_spa = torch.cat((x_spa, spa_cls_tokens), dim=1)

        x_spe = self.spe_patch_embed(x[:,:,self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1,
                                       self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1])
        x_spe = x_spe + self.spe_pos_embed[:, :-1, :]
        # append cls token
        if self.cls:
          spe_cls_token = self.spe_cls_token + self.spe_pos_embed[:, -1:, :]
          spe_cls_tokens = spe_cls_token.expand(x_spe.shape[0], -1, -1)
          x_spe = torch.cat((x_spe, spe_cls_tokens), dim=1)

        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)
        if self.global_pool:
            x_spa = x_spa[:, 0:self.cs, :].mean(dim=1)  # global pool without cls token
            x_spe = x_spe[:, 0:self.cs, :].mean(dim=1)
            outcome = self.fc_norm((x_spa + x_spe)/2)
        else:
            outcome = (x_spa[:, -1] + x_spe[:, -1])/2
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        out = self.head(x)  
            
        return out

