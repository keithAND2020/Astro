import torch.nn as nn
seed = 0,
train = dict(max_epoch=100,
             ddp=True,
             optimizer=dict(
                 type='Adam',
                 lr=0.002,
                 betas=(0.9, 0.999),),
             ),
model = dict(type='SwinIR',
             img_size=256, patch_size=1, in_chans=1, out_chans=1,
             embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
             window_size=8, mlp_ratio=2., upscale=1, img_range=1.,
             upsampler='pixelshuffledirect', resi_connection='1conv', scale=1,
             ),
# model = dict(type='Restormer',inp_channels=1,out_channels=1,dim=48,num_blocks=[4, 6, 6, 8],
#              num_refinement_blocks=4,heads=[1, 2, 4, 8],ffn_expansion_factor=2.66,
#              bias=True,LayerNorm_type='WithBias',dual_pixel_task=False),
# model = dict(type='HAT',img_size=256, patch_size=1, in_chans=1, embed_dim=96, depths=(6, 6, 6, 6),
#             num_heads=(6, 6, 6, 6),window_size=8,compress_ratio=3,squeeze_factor=30,
#             conv_scale=0.01,overlap_ratio=0.5,mlp_ratio=4., qkv_bias=True,
#             upscale=1,img_range=1.,upsampler='pixelshuffle',resi_connection='1conv',),

# model = dict(type='EDSR'),
# model = dict(type='RCAN'),


dataset = dict(type='RFI_MultiInput',
               batch_size=1,
               num_workers=16,
               root_dir='/home/bingxing2/ailab/scxlab0063/AstroIR/dataset/',

               )
