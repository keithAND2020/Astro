import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from skimage import data
from skimage import color
from skimage.util import view_as_blocks
import scipy as sp
from scipy.fftpack import dct, dctn, idctn
from skimage.util.shape import view_as_windows, view_as_blocks
from PIL import Image
from astropy.visualization import ZScaleInterval
from matplotlib.colors import Normalize
from PIL import Image
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import data, img_as_float


def psnrhvsm(img1=None, img2=None, wstep=8, *args, **kwargs):
    varargin = args
    nargin = 2 + len(varargin)
    p_hvs_m = 0
    p_hvs = 0

    if nargin < 2:
        p_hvs_m = - Inf
        p_hvs = - Inf
        print('returned on narngin')
        return p_hvs_m, p_hvs

    if img1.size != img2.size:
        p_hvs_m = - Inf
        p_hvs = - Inf

        print('returned on img seizes')
        return p_hvs_m, p_hvs

    if nargin > 2:
        step = wstep
    else:
        step = 8

    LenXY = img1.shape
    (LenX, LenY) = LenXY
    print('lenx: {:.0f}, leny: {:.0f}'.format(LenX, LenY))

    CSFCof = np.array([[1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
                       [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
                       [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
                       [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
                       [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
                       [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
                       [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
                       [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.25995]])
    # see an explanation in [2]

    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
                        [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
                        [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
                        [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
                        [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
                        [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
                        [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
                        [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])
    # see an explanation in [1]

    S1 = 0
    S2 = 0
    Num = 0
    window_shape = (8, 8)

    A = view_as_blocks(img1, window_shape)
    B = view_as_blocks(img2, window_shape)
    print('A shape:', A.shape)
    num_patchsA = A.shape[0]
    num_patchsB = B.shape[0]
    for p in range(num_patchsA):
        for py in range(num_patchsB):
            # compute the 2d Discrete Cosine Transform
            patchA = A[p][py]
            patchB = B[p][py]
            # dct2(A)
            a_dct = dct(patchA, type=2, axis=1, norm='ortho')
            A_dct = dct(a_dct, type=2, axis=0, norm='ortho')
            # dct2(B)
            b_dct = dct(patchB, type=2, axis=1, norm='ortho')
            B_dct = dct(b_dct, type=2, axis=0, norm='ortho')
            MaskA = maskeff(patchA, A_dct)
            MaskB = maskeff(patchB, B_dct)
            if MaskB > MaskA:
                MaskA = MaskB.copy()
            for k in range(7):
                for l in range(7):
                    u = abs(A_dct[k, l] - B_dct[k, l])
                    S2 = S2 + ((np.dot(u, CSFCof[k, l])) ** 2)  # PSNR-hvs
                    if (k != 1) or (l != 1):
                        if u < MaskA / MaskCof[k, l]:
                            u = 0
                        else:
                            u = u - (MaskA / MaskCof[k, l])
                    S1 = S1 + ((np.dot(u, CSFCof[k, l])) ** 2)  # PSNR-HVS-M
                    Num = Num + 1

    if Num != 0:
        S1 = S1 / Num
        S2 = S2 / Num
        if S1 == 0:
            p_hvs_m = 100000
        else:
            p_hvs_m = 10 * (np.log10(255 * (255 / S1)))
            print('p_hvs_m: {:.0f}'.format(p_hvs_m))
        if S2 == 0:
            p_hvs = 100000
        else:
            p_hvs = 10 * (np.log10(255 * (255 / S2)))
            print('p_hvs: {:.0f}'.format(p_hvs))
    return p_hvs_m, p_hvs

def maskeff(z=None, zdct=None, *args, **kwargs):
    # Calculation of Enorm value (see [1])
    m = 0
    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
                        [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
                        [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
                        [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
                        [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
                        [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
                        [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
                        [0.01929, 0.0118150, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])
    # see an explanation in [1]
    for k in range(7):
        for l in range(7):
            if (k != 1) or (l != 1):
                m = m + np.dot((zdct[k, l] ** 2), MaskCof[k, l])

    pop = vari(z)
    if pop != 0:
        block1 = vari(z[0:3, 0:3])
        block2 = vari(z[0:3, 4:7])
        block3 = vari(z[4:7, 4:7])
        block4 = vari(z[4:7, 0:3])
        pop = (block1 + block2 + block3 + block4) / pop

    m = np.sqrt(np.dot(m, pop)) / 32
    return m

def vari(AA=None, *args, **kwargs):
    varargin = args
    flat = AA.flatten(order='C')
    varia = np.var(flat)
    d = np.dot(varia, flat.size)
    return d

def fits_vis(ori_array):
    z = ZScaleInterval(n_samples=1000, contrast=0.25)
    z1, z2 = z.get_limits(ori_array)  # 19个一起统计中位数 、 方差
    norm = Normalize(vmin=z1, vmax=z2)
    normalized_array = norm(ori_array)
    cmap = plt.get_cmap('gray')
    wave_array = cmap(normalized_array)
    wave_array = (wave_array[..., 0] * 255).astype(np.uint8)
    return wave_array.astype(np.uint8)

def cal_score(img):
    # img = fits_vis(np.array(img))
    img = np.log(np.array(img))
    imask = img[0]
    noise = img[1]
    p1, p2 = psnrhvsm(noise,imask,wstep=8)
    print('p_hvs_m: {:.0f} dB'.format(p1))
    # print('p_hvs: {:.0f} dB'.format(p2))
    return p1, imask, noise


def plot_two_arrays(array1, array2, psnr, hvsm):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(array1, cmap='viridis')
    axs[0].set_title('target')
    axs[0].axis('off')  # 可选：关闭坐标轴
    axs[1].imshow(array2, cmap='viridis')
    axs[1].set_title('source')
    axs[1].axis('off')  # 可选：关闭坐标轴
    fig.text(0.5, 0.01, f'PSNRHVSM:{hvsm}', ha='center')  # 第一行文字，靠下中间对齐
    fig.text(0.5, 0.04, f'PSNR:{psnr}', ha='center')  # 第二行文字，稍微往上一点
    plt.tight_layout()
    plt.savefig(f"figs/{hvsm}.png", bbox_inches='tight')


# 图像路径
# dir_path = "/ailab/group/pjlab-ai4s/ai4astro/Deep_space_explore/patchify/sdss2hst_new/1237664093441818802-hst_skycell-p2039x08y03_wfc3_uvis_f814w_all_drc"
dir_path = "eval_data/"
data_list = glob.glob(dir_path + "*.npy")

score_list = []
for i in tqdm(data_list):
    data = np.load(i, allow_pickle=True)
    img_1 = data.item().get('target') * data.item().get('mask')
    img_1 = np.nan_to_num(img_1)
    img_2 = data.item().get('source')[:, :, 2] * data.item().get('mask')
    img_2 = np.nan_to_num(img_2)
    img = np.stack((img_1, img_2), axis=0)
    score, imask, noise = cal_score(img)
    score_list.append(score)

    img1_float = img_as_float(img_1)
    img2_float = img_as_float(img_2)
    psnr_value = psnr(img1_float, img2_float, data_range=img1_float.max() - img1_float.min())

    plot_two_arrays(imask, noise, psnr_value, score)





