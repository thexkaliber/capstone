import os
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import torch
from astropy.table import Table
from dinov2.eval.setup import setup_and_build_model
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from astroclip.astrodino.utils import setup_astrodino
from astroclip.models import AstroClipModel, Moco_v2, SpecFormer
from astroclip.models.astroclip import ImageHead, ImageHeadSwin, SpectrumHead

import os
from argparse import ArgumentParser
from typing import Dict
import numpy as np
import torch
from astropy.table import Table
from dinov2.eval.setup import setup_and_build_model
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm
from astroclip.astrodino.utils import setup_astrodino
from astroclip.models import AstroClipModel, Moco_v2, SpecFormer
from astroclip.models.astroclip import ImageHead, ImageHeadSwin, SpectrumHead
import h5py
from concurrent.futures import ThreadPoolExecutor


def get_embeddings(
     image_models: Dict[str, torch.nn.Module],
     spectrum_models: Dict[str, torch.nn.Module],
     images: torch.Tensor,
     spectra: torch.Tensor,
     batch_size: int = 512,
 ) -> dict:
     """Get embeddings for images using models"""
     full_keys = set(image_models.keys()).union(spectrum_models.keys())
     model_embeddings = {key: [] for key in full_keys}
     im_batch, sp_batch = [], []
     device = "cuda:1"
     assert len(images) == len(spectra)
     for image, spectrum in tqdm(zip(images, spectra)):
         # Load images, already preprocessed
         im_batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])
         sp_batch.append(torch.tensor(spectrum, dtype=torch.float32)[None, :, :])

         # Get embeddings for batch
         if len(im_batch) == batch_size:
             with torch.no_grad():
                 spectra, images = torch.cat(sp_batch).cuda().to(device), torch.cat(im_batch).cuda().to(device)

                 for key in image_models.keys():
                     model_embeddings[key].append(image_models[key](images))

                 for key in spectrum_models.keys():
                     model_embeddings[key].append(spectrum_models[key](spectra))

             im_batch, sp_batch = [], []

     # Get embeddings for last batch
     if len(im_batch) > 0:
         with torch.no_grad():
             spectra, images = torch.cat(sp_batch).cuda().to(device), torch.cat(im_batch).cuda().to(device)

             # Get embeddings
             for key in image_models.keys():
                 model_embeddings[key].append(image_models[key](images))

             for key in spectrum_models.keys():
                 model_embeddings[key].append(spectrum_models[key](spectra))

     model_embeddings = {
         key: np.concatenate(model_embeddings[key]) for key in model_embeddings.keys()
     }
     return model_embeddings


def embed_provabgs(
    provabgs_file_train: str,
    provabgs_file_test: str,
    pretrained_dir: str,
    batch_size: int = 512,
    device: str = "cuda:1",
    num_workers: int = 16,
):
    # Get directories
    astrodino_output_dir = os.path.join(pretrained_dir, "astrodino_output_dir")
    
    pretrained_weights = {
    "astroclip" : os.path.join(pretrained_dir, "astroclip.ckpt"), 
    "specformer": os.path.join(pretrained_dir, "specformer.ckpt"),
    "astrodino": os.path.join(pretrained_dir, "astrodino.ckpt"), 
    "xcit_encoder": os.path.join(pretrained_dir, "xcit_encoder.pth"),
    "swin_encoder": os.path.join(pretrained_dir, "swin_encoder.pth"),
    "swinv2_encoder": os.path.join(pretrained_dir, "swinv2_encoder.pth"),
    "vim_encoder": os.path.join(pretrained_dir, "vim_encoder.pth"),
    "astroclip_xcit" : os.path.join(pretrained_dir, "astroclip_xcit.ckpt"), 
    "astroclip_swin" : os.path.join(pretrained_dir, "astroclip_swin.ckpt"), 
    "astroclip_swinv2" : os.path.join(pretrained_dir, "astroclip_swinv2.ckpt"), 
    "astroclip_vim" : os.path.join(pretrained_dir, "astroclip_vim.ckpt"), 
    }
    
    # # Set up Stein, et al. model
    # stein = Moco_v2.load_from_checkpoint(
    #     checkpoint_path=pretrained_weights["stein"],
    # ).encoder_q

    #Set up SpecFormer model
    checkpoint = torch.load(pretrained_weights["specformer"])
    specformer = SpecFormer(**checkpoint["hyper_parameters"])
    specformer.load_state_dict(checkpoint["state_dict"])
    specformer.cuda().to(device)
    print("specformer setup...")

    #Set up AstroDINO model
    astrodino = setup_astrodino(astrodino_output_dir, pretrained_weights["astrodino"], "/workspace/astroclip/src/astroclip/astroclip/astrodino/config.yaml","astrodino").to(device)
    print("astrodino setup...")
    xcit_encoder = setup_astrodino(astrodino_output_dir, pretrained_weights["xcit_encoder"], "/workspace/astroclip/src/astroclip/astroclip/models/xcit/config.yaml","xcit").to(device)
    print("xcit setup...")

    swin_encoder = setup_astrodino(astrodino_output_dir, pretrained_weights["swin_encoder"], "/workspace/astroclip/src/astroclip/astroclip/models/swin/config.yaml","swin").to(device)
    print("swin setup...")

    swinv2_encoder = setup_astrodino(astrodino_output_dir, pretrained_weights["swinv2_encoder"], "/workspace/astroclip/src/astroclip/astroclip/models/swin/config.yaml","swinv2").to(device)
    print("swinv2 setup...")

    vim_encoder = setup_astrodino(astrodino_output_dir, pretrained_weights["vim_encoder"], "/workspace/astroclip/src/astroclip/astroclip/models/vim/config.yaml","vim").to(device)
    print("vim setup...")

    #Set up AstroCLIP
    astroclip = AstroClipModel.load_from_checkpoint(
         checkpoint_path=pretrained_weights["astroclip"],
    ).to(device)
    print("astroclip setup...")

    xcit_head = ImageHead(model_name="xcit", config="/workspace/astroclip/src/astroclip/astroclip/models/xcit/config.yaml",model_weights=pretrained_weights["xcit_encoder"], save_directory="/workspace/astroclip/outputs")
    swin_head = ImageHeadSwin(model_name="swin", config="/workspace/astroclip/src/astroclip/astroclip/models/swin/config.yaml",model_weights=pretrained_weights["swin_encoder"], save_directory="/workspace/astroclip/outputs")    
    swinv2_head = ImageHeadSwin(model_name="swinv2", config="/workspace/astroclip/src/astroclip/astroclip/models/swin/config.yaml",model_weights=pretrained_weights["swinv2_encoder"], save_directory="/workspace/astroclip/outputs")
    vim_head = ImageHead(model_name="vim", config="/workspace/astroclip/src/astroclip/astroclip/models/vim/config.yaml",model_weights=pretrained_weights["vim_encoder"], save_directory="/workspace/astroclip/outputs")


    spectrum_head = SpectrumHead(model_path=pretrained_weights["specformer"]).to(device)

    astroclip_xcit = AstroClipModel.load_from_checkpoint(
         checkpoint_path=pretrained_weights["astroclip_xcit"], image_encoder=xcit_head, spectrum_encoder=spectrum_head, 
    ).to(device)
    print("astroclip_xcit setup...")
    
    astroclip_swin = AstroClipModel.load_from_checkpoint(
         checkpoint_path=pretrained_weights["astroclip_swin"], image_encoder=swin_head, spectrum_encoder=spectrum_head, 
    ).to(device)
    print("astroclip_swin setup...")

    astroclip_swinv2 = AstroClipModel.load_from_checkpoint(
         checkpoint_path=pretrained_weights["astroclip_swinv2"], image_encoder=swinv2_head, spectrum_encoder=spectrum_head, 
    ).to(device)
    print("astroclip_swinv2 setup...")

    astroclip_vim = AstroClipModel.load_from_checkpoint(
        checkpoint_path=pretrained_weights["astroclip_vim"], image_encoder=vim_head, spectrum_encoder=spectrum_head, 
    ).to(device)
    print("astroclip_vim setup...")


    
    # Set up model dict
    image_models = {
        "astrodino": lambda x: astrodino(x).cpu().numpy(),
        "xcit_encoder": lambda x: xcit_encoder(x).cpu().numpy(),
        "swin_encoder": lambda x: swin_encoder(x).cpu().numpy(),
        "swinv2_encoder": lambda x: swinv2_encoder(x).cpu().numpy(),
        "vim_encoder": lambda x: vim_encoder(x).cpu().numpy(),
        "astroclip_image": lambda x: astroclip(x, input_type="image").cpu().numpy(),
        "astroclip_xcit": lambda x: astroclip_xcit(x, input_type="image").cpu().numpy(),
        "astroclip_swin": lambda x: astroclip_swin(x, input_type="image").cpu().numpy(),
        "astroclip_swinv2": lambda x: astroclip_swinv2(x, input_type="image").cpu().numpy(),
        "astroclip_vim": lambda x: astroclip_vim(x, input_type="image").cpu().numpy(),
        
    }

    spectrum_models = {
        "astroclip_spectrum": lambda x: astroclip(x, input_type="spectrum")
        .cpu()
        .numpy(),
        "astroclip_xcit_spectrum": lambda x: astroclip_xcit(x, input_type="spectrum").cpu().numpy(),
        "astroclip_swin_spectrum": lambda x: astroclip_swin(x, input_type="spectrum").cpu().numpy(),
        "astroclip_swinv2_spectrum": lambda x: astroclip_swinv2(x, input_type="spectrum").cpu().numpy(),
        "astroclip_vim_spectrum": lambda x: astroclip_vim(x, input_type="spectrum").cpu().numpy(),
        "specformer": lambda x: np.mean(
            specformer(x)["embedding"].cpu().numpy(), axis=1
        ),
    }
    print("Models are correctly set up!")

    # Load data
    files = [provabgs_file_test, provabgs_file_train]
    for f in files:
        provabgs = Table.read(f)
        images, spectra = provabgs["image"], provabgs["spectrum"]
        
        # Get embeddings
        embeddings = get_embeddings(
             image_models, spectrum_models, images, spectra, batch_size
        )

        # Remove images and replace with embeddings
        #provabgs.remove_column("image")
        #provabgs.remove_column("spectrum")
        for key in embeddings.keys():
             assert len(embeddings[key]) == len(provabgs), "Embeddings incorrect length"
             provabgs[f"{key}_embeddings"] = embeddings[key]

        # Save embeddings
        provabgs.write(f.replace(".hdf5", "_ssl_embeddings.hdf5"), overwrite=True)

if __name__ == "__main__":
    ASTROCLIP_ROOT = "/workspace/astroclip/"
    parser = ArgumentParser()
    parser.add_argument("--provabgs_file_train",type=str,default=os.path.join(ASTROCLIP_ROOT, "data/provabgs_paired_train.hdf5"),)
    parser.add_argument("--provabgs_file_test",type=str,default=os.path.join(ASTROCLIP_ROOT, "data/provabgs_paired_test.hdf5"),)
    parser.add_argument("--pretrained_dir",type=str,default=ASTROCLIP_ROOT,)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    embed_provabgs(
        args.provabgs_file_train,
        args.provabgs_file_test,
        args.pretrained_dir,
        args.batch_size,
        args.device,
        args.num_workers,
    )

