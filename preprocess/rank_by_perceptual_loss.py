from glob import glob
from os.path import basename
import lpips
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
loss_fn_vgg = lpips.LPIPS(net='vgg')

def load_img_tensor(img_path):
    im = Image.open(img_path)
    im = im.resize((256, 256))
    arr = torch.from_numpy(np.array(im)).movedim(2, 0)[None]
    return (((arr / 255) - 0.5) * 2)

styles = ['toonify', 'OR', 'BAO', 'EX0', 'EX3', 'EX5', 'IM1', 'P00', 'P04', 'P06', 'RE0', 'R00', 'R02']

files = []

for img_path in tqdm(sorted(glob('images/stills/*'))):
    bname = basename(img_path)
    img = []
    ref_arr = load_img_tensor(img_path)
    img.append(ref_arr)
    perceptual_diffs = []
    for style in styles:
        comp_arr = load_img_tensor(f'images/{style}/{bname}')
        img.append(comp_arr)
        d = loss_fn_vgg(ref_arr, comp_arr)
        perceptual_diffs.append(float(d[0][0][0][0].detach().numpy()))

    result = dict(zip(styles, perceptual_diffs))
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)

    sorted_keys = list(reversed([s[0] for s in sorted_result]))
    idx = [styles.index(k) for k in sorted_keys]
    img_reordered = [img[0]] + [img[i+1] for i in idx]

    print(sorted_keys)

    grid = make_grid(
        torch.cat(img_reordered[:5], 0),
        normalize=True,
        range=(-1, 1),
    )

    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save('top4_' + bname)

    files.extend([f'{dir}/{bname}' for dir in ['stills'] + sorted_keys[:4]])

with open('files.txt', 'w') as f:
    f.writelines([l + '\n' for l in files])