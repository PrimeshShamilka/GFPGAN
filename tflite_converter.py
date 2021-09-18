from gfpgan import GFPGANer
from realesrgan import RealESRGANer
import torch

# bg_upsampler = RealESRGANer(
#     scale=2,
#     model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
#     tile=400,
#     tile_pad=10,
#     pre_pad=0,
#     half=True)

model = GFPGANer(
    model_path='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

model.load_state_dict(torch.load('experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth', map_location='cpu')).eval()

