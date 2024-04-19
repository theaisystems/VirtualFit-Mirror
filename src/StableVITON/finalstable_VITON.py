import os

os.chdir(r'E:\AsadHussain\tryon\detectron2')

from detectron2.config import get_cfg
import time
from detectron2.engine import DefaultPredictor
os.chdir(r'E:\AsadHussain\tryon\detectron2\projects\DensePose2')
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer


from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
#os.chdir('/content/SelfCorrectionHumanParsing')
#import networks 
#from SelfCorrectionHumanParsing.networks import networks

#from utils2.transforms import transform_logits
#from datasets.simple_extractor_dataset import SimpleFolderDataset
os.chdir(r'E:\AsadHussain\tryon\Cloth_segmentation')
from network import U2NET

import matplotlib.pyplot as plt


import numpy as np
import torch.nn.functional as F

from collections import OrderedDict
from options import opt
os.chdir(r'E:\AsadHussain\tryon\StableVITON')

from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse
import shutil
import cv2
import torch
from torch.utils.data import DataLoader
from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img

#class to normalize images for the segmentation model
class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"

#os.chdir('/content/StableVITON')


#model_restore = r'/content/SelfCorrectionHumanParsing/checkpoints/final.pth'


# detectron output

def detectron(INPUT_IMAGE_PATH, predictor):
  # cfg = get_cfg()
  # add_densepose_config(cfg)
  # cfg.merge_from_file(r"E:\AsadHussain\tryon\detectron2\projects\DensePose2\configs\densepose_rcnn_R_50_FPN_s1x_legacy.yaml")
  # cfg.MODEL.WEIGHTS = r"E:\AsadHussain\tryon\detectron2\projects\DensePose2\DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl"
  # predictor = DefaultPredictor(cfg)
  image = cv2.imread(INPUT_IMAGE_PATH)
  #image = cv2.resize(img,(512,512))
  height, width = image.shape[:2]
  t1 = time.time()
  # Process the image
  with torch.no_grad():

      outputs = predictor(image)['instances']

  results = DensePoseResultExtractor()(outputs)
  t2 = time.time()
  print(f'{t2-t1} sec')
  cmap = cv2.COLORMAP_VIRIDIS

  # Visualizer outputs black for background, but we want the 0 value of
  # the colormap, so we initialize the array with that value
  arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
  output_image = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
  # return output_image
  # Save the output image
  #cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
  return output_image

#segmentation output

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette






def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)



def generate_mask(input_image, net, palette, device = 'cuda'):
    #img = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)
    img = cv2.imread(input_image)
    img_size = img.shape[0], img.shape[1]
    img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_CUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    #alpha_out_dir = os.path.join(opt.output, 'alpha')
    #cloth_seg_out_dir = os.path.join(opt.output, 'cloth_seg')

    #os.makedirs(alpha_out_dir, exist_ok=True)
    #os.makedirs(cloth_seg_out_dir, exist_ok=True)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes_to_save = []

    for cls in range(1, 4):
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    # for cls in classes_to_save:
    #     alpha_mask = (output_arr == cls).astype(np.uint8) * 255
    #     alpha_mask = alpha_mask[0]
    #     alpha_mask = cv2.resize(alpha_mask, img_size[::-1], interpolation=cv2.INTER_CUBIC)
    #     cv2.imwrite(os.path.join(alpha_out_dir, f'{cls}.png'), alpha_mask)

    cloth_seg = np.zeros((*output_arr[0].shape, 3), dtype=np.uint8)
    for cls in range(1, 4):
        cloth_seg[output_arr[0] == cls] = palette[3 * cls:3 * (cls + 1)]

    cloth_seg = cv2.resize(cloth_seg, img_size[::-1], interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(os.path.join(cloth_seg_out_dir, 'final_seg.png'), cloth_seg)

    return cloth_seg





def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    #check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def get_upper_garment(img, img_parse_map):
    # Extract individual channels
    
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    red_channel = img_parse_map[:, :, 0]
    green_channel = img_parse_map[:, :, 1]
    blue_channel = img_parse_map[:, :, 2]

    # Create a mask where blue is 128 and red and green are 0
    mask = (red_channel == 128) & (green_channel == 0) & (blue_channel == 0)
    # Use the mask to extract the corresponding area from the original image
    upper_garment_segment = (mask.reshape(*mask.shape, 1) * img).astype(np.uint8)

    return upper_garment_segment
    

def generate_rgb_agnostic(img, img_agnostic_parse_map):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    sum_img_agnostic_parse_map = np.sum(img_agnostic_parse_map, axis = 2)
    sum_img_agnostic_parse_map[sum_img_agnostic_parse_map!=255] = 1
    img_rgb_agnostic = (sum_img_agnostic_parse_map.reshape(*sum_img_agnostic_parse_map.shape,1)*img).astype(np.uint8)
    return img_rgb_agnostic

def preprocess_images(img,model_pre, predictor): # img == pth of image file
    name = img.split('\\')[-1].split('.')[0]
    name_ext = img.split('\\')[-1]
    checkpoint_path = r'E:\AsadHussain\tryon\Cloth_segmentation\cloth_segm.pth'
    in_detectron = img
    out_detectron = rf'E:\AsadHussain\tryon\StableVITON\test\image-densepose\{name}.jpg'
    densepose = detectron(in_detectron, predictor)
    device = 'cpu'
    # Create an instance of your model
    #model_pre = load_seg_model(r"E:\AsadHussain\tryon\Cloth_segmentation\cloth_segm.pth")

    palette = get_palette(4)

    #img = Image.open(img).convert('RGB')

    cloth_seg = generate_mask(img, net=model_pre, palette=palette, device=device)
    plt.imshow(cloth_seg)
    plt.title("segmente bef")
    plt.show()
    
    upper = get_upper_garment(img,cloth_seg)
    upper_garm = upper.copy()
    upper_garm[np.all(upper_garm != (0, 0, 0), axis=-1)] = (255,255,255)
    plt.imshow(upper_garm)
    plt.title("uppergarment bef")
    plt.show()
   # upper_garment_file = rf'E:\AsadHussain\tryon\StableVITON\test\gnostic-mask\{name}maskmaskmask.png'
    #cv2.imwrite(upper_garment_file, upper_garm)

    rgb_agn_init = generate_rgb_agnostic(img, cloth_seg)
    rgb_agn =  rgb_agn_init - upper
    rgb_agn[np.all(rgb_agn == (0, 0, 0), axis=-1)] = (128,128,128)
    plt.imshow(rgb_agn)
    plt.title("rgbagnostic bef")
    plt.show()
    #rgbagnostic_file = rf'E:\AsadHussain\tryon\StableVITON\test\gnostic-v3.2\{name}-agnosticagnostic.jpg'
    #cv2.imwrite(rgbagnostic_file, cv2.cvtColor(rgb_agn, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(rgb_agn, cv2.COLOR_BGR2RGB), upper_garm, densepose

# def create_txt(image, cloth):
#   f = open(r"E:\AsadHussain\tryon\StableVITON\test_pairs.txt", "w")
#   text = f'{image} {cloth}'
#   f.write(text)
#   f.close()

# the setting of model
batch_size = 1
img_H = 512
img_W = 384
config_path = r"E:\AsadHussain\tryon\StableVITON\configs\VITON512.yaml"
model_path = r"E:\AsadHussain\tryon\StableVITON\VITONHD.ckpt"
config = OmegaConf.load(config_path)
config.model.params.img_H = img_H
config.model.params.img_W = img_W
params = config.model.params
unpair = True
save_dir = r'E:\AsadHussain\tryon\StableVITON\Output'
data_root_dir = r'E:\AsadHussain\tryon\StableVITON'
eta = 0.0
repaint = True
denoise_steps = 20

#segmentation model
model_pre = load_seg_model(r"E:\AsadHussain\tryon\Cloth_segmentation\cloth_segm.pth")

#desnsepose model
cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(r"E:\AsadHussain\tryon\detectron2\projects\DensePose2\configs\densepose_rcnn_R_50_FPN_s1x_legacy.yaml")
cfg.MODEL.WEIGHTS = r"E:\AsadHussain\tryon\detectron2\projects\DensePose2\DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl"
predictor = DefaultPredictor(cfg)

#stableViTon Model
model = create_model(config_path=None, config=config)
model = model.cuda()
model.load_state_dict(torch.load(model_path, map_location="cpu"))

model.eval()

def final_model(model, model_pre, predictor, img, cloth_pth, save_dir=save_dir):
    
    # cloth_image = cv2.cvtColor(cv2.imread(cloth_pth),cv2.COLOR_BGR2RGB)
    cloth_pth = rf'E:\AsadHussain\tryon\cloth\{cloth_pth}'
    cloth_image = cv2.imread(cloth_pth)
    
    databytes = np.asarray(bytearray(img), dtype="uint8")
    image = cv2.imdecode(databytes,cv2.IMREAD_COLOR)
    img_pth = r"E:\AsadHussain\tryon\StableVITON\test\middle\input.jpg"
    cv2.imread(img_pth, image)
    image = cv2.imread(img_pth)
    
    agnostic, agnostic_mask, densepose = preprocess_images(img_pth,model_pre,predictor)
    
    sampler = PLMSSampler(model)
    dataset = getattr(import_module("dataset2"), config.dataset_name)(
        data_root_dir=data_root_dir,
        img_H=img_H,
        img_W=img_W,
        image=image,
        cloth_image=cloth_image,
        agnostic=agnostic,
        agnostic_mask=agnostic_mask,
        densepose=densepose,
        is_paired=not unpair,
        is_test=True,
        is_sorted=True
    )
    
    
    dataloader = DataLoader(dataset, num_workers=1, shuffle=False, batch_size=batch_size, pin_memory=True)
    
    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(save_dir, "unpair" if unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        z, c = model.get_input(batch, params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                #batch[k] = v.cuda() #!!! back to cuda when required
                batch[k] = v
        sampler.model.batch = batch
    
        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = model.q_sample(z, ts)     
    
        samples, _, _ = sampler.sample(
            denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=eta,
            unconditional_conditioning=uc_full,
        )
    
        x_samples = model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                cv2.imwrite(r"E:\AsadHussain\tryon\StableVITON\test\middle\repaint_mask_img.jpg", repaint_agn_img)
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)
    
            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}nenenenenene.jpg")
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])
    
    return x_sample_img[:,:,::-1]





# imagename = r"E:\AsadHussain\tryon\StableVITON\test\image\00484_00.jpg"
# # agnostic, agnostic_mask, densepose = preprocess_images(r"E:\AsadHussain\tryon\StableVITON\test\image\00484_00.jpg")

# create_txt(imagename.split('\\')[-1], '00012_00.jpg')

# image_pth = r"E:\AsadHussain\tryon\StableVITON\test\image\00484_00.jpg"
# # cloth_pth = r"E:\AsadHussain\tryon\StableVITON\test\cloth\00484_00.jpg"
# cloth_pth = r"E:\AsadHussain\tryon\cloth\00034_00.jpg"
def main_viton(image_pth, cloth_pth):
    finalimage = final_model(model,model_pre, predictor, image_pth, cloth_pth, r"E:\AsadHussain\tryon\StableVITON\Output\newnewn")
    cv2.imwrite(r"C:\Users\asad\Desktop\idharao.jpg",finalimage)
    return finalimage
    
