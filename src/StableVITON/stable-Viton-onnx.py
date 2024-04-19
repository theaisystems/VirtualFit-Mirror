import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse
import torch
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default=r"E:\AsadHussain\tryon\StableVITON\configs\VITON512.yaml", type=str)
parser.add_argument("--model_load_path", default=r"E:\AsadHussain\tryon\StableVITON\VITONHD.ckpt", type=str)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--data_root_dir", type=str, default=r"E:\AsadHussain\tryon\StableVITON")
parser.add_argument("--repaint", action="store_true")
parser.add_argument("--unpair", action="store_true")
parser.add_argument("--save_dir", type=str, default=r"E:\AsadHussain\tryon\StableVITON\Output")

parser.add_argument("--denoise_steps", type=int, default=20)
parser.add_argument("--img_H", type=int, default=512)
parser.add_argument("--img_W", type=int, default=384)
parser.add_argument("--eta", type=float, default=0.0)
args = parser.parse_args()
    # return args
onnx_path = r'E:\AsadHussain\tryon\StableVITON\stabl-viton.onnx'

with torch.no_grad():

# def main(args):
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    config = OmegaConf.load(r'E:\AsadHussain\tryon\StableVITON\configs\VITON512.yaml')
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params

    model = create_model(config_path=None, config=config)
    model.load_state_dict(torch.load(args.model_load_path, map_location="cpu"))
    #model = model.cuda()
    model.eval()
    output = model(torch.randn([1,256,256]))

with torch.inference_mode():

    sampler = PLMSSampler(model)
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        is_paired=not args.unpair,
        is_test=True,
        is_sorted=True
    )
    dataloader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"batchbatchbatchbatch{batch_idx}/{len(dataloader)}")
        print(batch)
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
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = model.q_sample(z, ts) 
    #print(c)
    #batch2 = {k:v[0].to('cpu') for k, v in batch.items() if isinstance(v[0], torch.Tensor)}
    #b2 = batch2[3]
        batch2 = [v[0].to('cpu') for k, v in batch.items() if isinstance(v[0], torch.Tensor)]
        t = torch.tensor([951],device='cpu')
        uc_full2 = {k: v.to('cpu') for k, v in batch.items() if isinstance(v, torch.Tensor)}
        dummy_input = (batch2,uc_full,t)
        torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11
                )
        print("CENDCENDCEND")
        
        model.to_onnx('E:\AsadHussain\tryon\StableVITON\stabl-viton.onnx',input_sample=model.get_sample_input)