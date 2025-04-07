_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/CFSR',visualize=True)
model = dict(type='CFSR',in_chans=1, embed_dim=48, depths=(6,6), dw_size=3,
                 mlp_ratio=2, upscale=2, img_range=1., upsampler='pixelshuffledirect'),
dataset = dict(type='SR_dataset',
               batch_size=16,
               num_workers=16,
               root_dir='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/dataset',
               filenames_file_train='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/dataload_filename/train_dataloader.txt',#'/home/bingxing2/ailab/group/ai4astro/Datasets/Astro_SR/train_dataloader.txt',
               filenames_file_eval='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/dataload_filename/train_dataloader.txt',#'/home/bingxing2/ailab/group/ai4astro/Datasets/Astro_SR/train_dataloader.txt',
               
               )
#CUDA_VISIBLE_DEVICES=0 bash tools/dist_trainval.sh configs/models/restormer.py