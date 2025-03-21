_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=1e-4))
model = dict(type='SwinIR',
             img_size=128, patch_size=1, in_chans=1, out_chans=1,
             embed_dim=90, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
             window_size=8, mlp_ratio=2., upscale=2, img_range=1.,
             upsampler='pixelshuffledirect', resi_connection='1conv'
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


dataset = dict(type='SR_dataset',
               batch_size=8,
               num_workers=6,
               root_dir='/ailab/user/wuguocheng/AstroIR/tools/creat_dataset/new_create_dataset/train_patches',
               filenames_file_train='/ailab/user/wuguocheng/Astro_SR/dataload_filename/trainfile.txt'
               )
