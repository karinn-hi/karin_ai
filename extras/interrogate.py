import os
import torch
import ldm_patched.modules.model_management as model_management
import zipfile

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules.model_loader import load_file_from_url
from modules.config import path_clip_vision, path_checkpoints
from ldm_patched.modules.model_patcher import ModelPatcher
from extras.BLIP.models.blip import blip_decoder
from deepdanbooru_tagger import DeepDanbooruTagger
from PIL import Image

blip_image_eval_size = 384
blip_repo_root = os.path.join(os.path.dirname(__file__), 'BLIP')

class BlipInterrogator:
    def __init__(self):
        self.blip_model = None
        self.load_device = torch.device('cpu')
        self.offload_device = torch.device('cpu')
        self.dtype = torch.float32

    @torch.no_grad()
    @torch.inference_mode()
    def interrogate(self, img_rgb):
        if self.blip_model is None:
            filename = load_file_from_url(
                url='https://huggingface.co/lllyasviel/misc/resolve/main/model_base_caption_capfilt_large.pth',
                model_dir=path_clip_vision,
                file_name='model_base_caption_capfilt_large.pth',
            )

            model = blip_decoder(pretrained=filename, image_size=blip_image_eval_size, vit='base',
                                 med_config=os.path.join(blip_repo_root, "configs", "med_config.json"))
            model.eval()

            self.load_device = model_management.text_encoder_device()
            self.offload_device = model_management.text_encoder_offload_device()
            self.dtype = torch.float32

            model.to(self.offload_device)

            if model_management.should_use_fp16(device=self.load_device):
                model.half()
                self.dtype = torch.float16

            self.blip_model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)

        model_management.load_model_gpu(self.blip_model)

        gpu_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(img_rgb).unsqueeze(0).to(device=self.load_device, dtype=self.dtype)

        caption = self.blip_model.model.generate(gpu_image, sample=True, num_beams=1, max_length=75)[0]

        return caption

class DeepDanbooruInterrogator:
    def __init__(self):
        self.tagger = None

    def load_model(self):
        model_zip_path = load_file_from_url(
            url='https://github.com/KichangKim/DeepDanbooru/releases/download/v4-20200814-sgd-e30/deepdanbooru-v4-20200814-sgd-e30.zip',
            model_dir=path_checkpoints,
            file_name='deepdanbooru-v4-20200814-sgd-e30.zip',
        )
        model_folder = os.path.join(path_checkpoints, 'deepdanbooru-v4-20200814-sgd-e30')
        if not os.path.exists(model_folder):
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(path_checkpoints)
        
        model_path = os.path.join(model_folder, 'model-resnet_custom_v4.h5')
        tags_path = os.path.join(model_folder, 'tags.txt')
        self.tagger = DeepDanbooruTagger(model_path, tags_path)

    @torch.no_grad()
    @torch.inference_mode()
    def interrogate(self, img_rgb):
        if self.tagger is None:
            self.load_model()

        pil_image = Image.fromarray(img_rgb)
        tags_dict = self.tagger.interrogate_image(pil_image, threshold=0.5)
        tags = [tag for tag, score in tags_dict.items()]
        return ', '.join(tags)

default_interrogator = BlipInterrogator().interrogate
default_anime_interrogator = DeepDanbooruInterrogator().interrogate