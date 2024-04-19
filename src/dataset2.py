from os.path import join as opj

import cv2
import numpy as np
from torch.utils.data import Dataset

def imread(p, h, w, is_mask=False, in_inverse_mask=False, img=None):
    # if img is None:
    #     img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        #img = p
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            image,
            cloth_image,
            agnostic,
            agnostic_mask,
            densepose,
            is_paired=True, 
            is_test=False, 
            is_sorted=False,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.image = image
        self.cloth_image = cloth_image
        self.agnostic  =agnostic
        self.agnostic_mask = agnostic_mask
        self.densepose = densepose
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
       
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        print("----------------------------")
        print(f'{self.drd}\{self.data_type}\gnostic-v3.2\{self.im_names[idx]}')
        ##agn = imread(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), self.img_H, self.img_W)
        #agn = imread(f'{self.drd}\{self.data_type}\gnostic-v3.2\{self.im_names[idx]}', self.img_H, self.img_W)
        agn = imread(self.agnostic, self.img_H, self.img_W)
        cv2.imwrite(r"E:\AsadHussain\tryon\StableVITON\Output\middle\agn.jpg",(agn+1)/2 * 255)
        ##agn_mask = imread(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), self.img_H, self.img_W, is_mask=True, in_inverse_mask=True)
        #agn_mask = imread(f'{self.drd}\{self.data_type}\gnostic-mask\{self.im_names[idx].replace(".jpg", "_mask.png")}', self.img_H, self.img_W, is_mask=True, in_inverse_mask=True)
        agn_mask = imread(self.agnostic_mask, self.img_H, self.img_W, is_mask=True, in_inverse_mask=True)
        cv2.imwrite(r"E:\AsadHussain\tryon\StableVITON\Output\middle\agn_mask.jpg",(agn_mask+1)/2 * 255)
        ##cloth = imread(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), self.img_H, self.img_W)
        cloth = imread(self.cloth_image, self.img_H, self.img_W)
        cv2.imwrite(r"E:\AsadHussain\tryon\StableVITON\Output\middle\cloth.jpg",(cloth+1)/2 * 255)
        ##image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
        #image = imread(f'{self.drd}\{self.data_type}\image\{self.im_names[idx]}', self.img_H, self.img_W)
        image = imread(self.image, self.img_H, self.img_W)
        cv2.imwrite(r"E:\AsadHussain\tryon\StableVITON\Output\middle\image.jpg",(image+1)/2 * 255)
        ##image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)
        #image_densepose = imread(f'{self.drd}\{self.data_type}\image-densepose\{self.im_names[idx]}', self.img_H, self.img_W)
        image_densepose = imread(self.densepose, self.img_H, self.img_W)
        cv2.imwrite(r"E:\AsadHussain\tryon\StableVITON\Output\middle\denspose.jpg",(image_densepose+1)/2 * 255)
        return dict(
            agn=agn,
            agn_mask=agn_mask,
            cloth=cloth,
            image=image,
            image_densepose=image_densepose,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
        )