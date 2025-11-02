import os
import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F
import numpy as np
from multiprocessing.managers import BaseManager
import hashlib
import io
import argparse
from filelock import FileLock

from geo_aware.utils.utils_correspondence import resize
from geo_aware.model_utils.extractor_sd import load_model, process_features_and_mask
from geo_aware.model_utils.extractor_dino import ViTExtractor
from geo_aware.model_utils.projection_network import AggregationNetwork

from utils.timer_utils import Timer
timer = Timer()

num_patches, image_size = 60, 480
primary_gpu, secondary_gpu = "cuda:0", "cuda:1"
aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device=primary_gpu)
aggre_net.load_pretrained_weights(torch.load('/home/exx/Projects/GeoAware-SC/results_spair/best_856.PTH'))
sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device=primary_gpu)


class GeoAware:
    def __init__(self):
        self.source_image, self.target_image = None, None
        self.source_feat, self.target_feat = None, None
        self.source_crop, self.target_crop = None, None
        self.cos = nn.CosineSimilarity(dim=1)
        self.max_cache_size = 10000

    def get_processed_features(self, image, text=None, cache_path=None, refresh_cache=False):
        try:
            desc_cache_path, cache_hit = None, False
            if cache_path is not None:
                desc_cache_path = cache_path.with_suffix(".desc.pt")

            if desc_cache_path is not None:
                with FileLock(str(desc_cache_path.with_suffix(".lock"))):
                    if desc_cache_path.exists() and not refresh_cache:
                        try:
                            desc = torch.load(desc_cache_path).to(primary_gpu)
                            cache_hit = True
                            return desc, cache_hit
                        except Exception as e:
                            print(e)
                            pass

            img_sd_input = resize(image, target_res=num_patches*16, resize=True, to_pil=True)
            features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, input_text=text, mask=False, raw=True)
            del features_sd['s2']

            img_dino_input = resize(image, target_res=num_patches*14, resize=True, to_pil=True)
            img_batch = extractor_vit.preprocess_pil(img_dino_input)
            features_dino = extractor_vit.extract_descriptors(img_batch.to(primary_gpu), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

            desc_gathered = torch.cat([
                    features_sd['s3'],
                    F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
                    F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
                    features_dino
                ], dim=1)
            desc = aggre_net(desc_gathered)
            norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
            desc = desc / (norms_desc + 1e-8)

            if desc_cache_path is not None:
                with FileLock(str(desc_cache_path.with_suffix(".lock"))):
                    try:
                        torch.save(desc, desc_cache_path)
                    except Exception as e:
                        print(e)
                        pass
            return desc, cache_hit
        except Exception as e:
            print(f"Error in get_processed_features: {e}")
    
    def load_images(self, source_path, target_path, source_crop=None, target_crop=None, cache_path=None, refresh_cache=False):
        timer.start("preprocess")
        self.source_image_original = Image.open(source_path).convert('RGB')
        self.target_image_original = Image.open(target_path).convert('RGB')
        if source_crop is not None:
            self.source_image_original = self.source_image_original.crop((
                int(source_crop[0]), int(source_crop[1]),
                self.source_image_original.size[0] - int(source_crop[2]), self.source_image_original.size[1] - int(source_crop[3])
            ))
        if target_crop is not None:
            self.target_image_original = self.target_image_original.crop((
                int(target_crop[0]), int(target_crop[1]),
                self.target_image_original.size[0] - int(target_crop[2]), self.target_image_original.size[1] - int(target_crop[3])
            ))
        self.source_image = resize(self.source_image_original, target_res=image_size, resize=True, to_pil=True)
        self.target_image = resize(self.target_image_original, target_res=image_size, resize=True, to_pil=True)
        self.source_crop = source_crop
        self.target_crop = target_crop
        source_cache_path, target_cache_path = None, None
        if cache_path is not None:
            with io.BytesIO() as buffer:
                self.source_image.save(buffer, format="PNG")
                source_cache_name = str(hashlib.sha256(buffer.getvalue()).hexdigest()[:16])
                source_cache_path = cache_path / source_cache_name
            with io.BytesIO() as buffer:
                self.target_image.save(buffer, format="PNG")
                target_cache_name = str(hashlib.sha256(buffer.getvalue()).hexdigest()[:16])
                target_cache_path = cache_path / target_cache_name
        timer.end("preprocess")
        with torch.no_grad():
            timer.start("get_processed_features_source")
            self.source_feat, source_cache_hit = self.get_processed_features(self.source_image, cache_path=source_cache_path, refresh_cache=refresh_cache)
            timer.end("get_processed_features_source")
            timer.start("get_processed_features_target")
            self.target_feat, target_cache_hit = self.get_processed_features(self.target_image, cache_path=target_cache_path, refresh_cache=refresh_cache)
            timer.end("get_processed_features_target")
            timer.start("upsample")
            self.source_feat_upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear')(self.source_feat).to(secondary_gpu)  # 1, C, H, W
            self.target_feat_upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear')(self.target_feat).to(secondary_gpu)  # 1, C, H, W
            timer.end("upsample")
        
        timer.start("delete_cache")
        if cache_path is not None:
            with FileLock(str(cache_path.with_suffix(".lock"))):
                cache_files = list(cache_path.glob("*"))
                if len(cache_files) > self.max_cache_size:
                        cache_files.sort(key=lambda f: f.stat().st_atime)
                        num_to_delete = len(cache_files) - self.max_cache_size
                        for f in cache_files[:num_to_delete]:
                            try:
                                f.unlink()
                            except Exception as e:
                                print(f"Error deleting cache file {f}: {e}")
        timer.end("delete_cache")

        return source_cache_hit, target_cache_hit

    def get_feature(self, image_path, crop=None, cache_path=None, refresh_cache=False):
        image = Image.open(image_path).convert('RGB')
        if crop is not None:
            image = image.crop((
                int(crop[0]), int(crop[1]),
                self.source_image_original.size[0] - int(crop[2]), self.source_image_original.size[1] - int(crop[3])
            ))
        image = resize(image, target_res=image_size, resize=True, to_pil=True)
        image_cache_path = None
        if cache_path is not None:
            with io.BytesIO() as buffer:
                image.save(buffer, format="PNG")
                cache_name = str(hashlib.sha256(buffer.getvalue()).hexdigest()[:16])
                image_cache_path = cache_path / cache_name
        with torch.no_grad():
            feat, cache_hit = self.get_processed_features(image, cache_path=image_cache_path, refresh_cache=refresh_cache)
        return feat.cpu().numpy(), cache_hit

    def compute_correspondence(self, x, y, target_mask=None):
        print("Computing correspondence...")
        timer.start("compute_correspondence")

        if self.source_crop is not None:
            x = x - self.source_crop[0]
            y = y - self.source_crop[1]
        x, y = self.original_to_resized(x, y, self.source_image_original.size[0], self.source_image_original.size[1])
        if x < 0 or y < 0 or x >= image_size or y >= image_size:
            raise ValueError(f"Coordinates ({x}, {y}) are out of bounds for image size {image_size}!")

        num_channel = self.source_feat_upsample.size(1)
        with torch.no_grad():
            src_vec = self.source_feat_upsample[0, :, y, x].view(1, num_channel, 1, 1)
            cos_map = self.cos(src_vec, self.target_feat_upsample)[0].cpu().numpy()
            if target_mask is not None:
                cos_map = cos_map * target_mask
            max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)
            xy = self.resized_to_original(max_yx[1], max_yx[0], self.target_image_original.size[0], self.target_image_original.size[1])
            score = cos_map[max_yx]
            if self.target_crop is not None:
                xy = (xy[0] + self.target_crop[0], xy[1] + self.target_crop[1])
            timer.end("compute_correspondence")
            return xy, score
    
    def get_source_info(self):
        return {
            "image_original": self.source_image_original,
            "image": self.source_image_original,
            "crop": self.source_crop,
        }
    
    def get_target_info(self):
        return {
            "image_original": self.target_image_original,
            "image": self.target_image_original,
            "crop": self.target_crop,
        }

    def original_to_resized(self, x, y, w, h):
        x = x + (max(w, h) - w) / 2
        y = y + (max(w, h) - h) / 2
        rescale = max(w, h) / 480
        x, y = x / rescale, y / rescale
        return int(x), int(y)
    
    def resized_to_original(self, x, y, w, h):
        rescale = max(w, h) / 480
        x, y = x * rescale, y * rescale
        x = x - (max(w, h) - w) / 2
        y = y - (max(w, h) - h) / 2
        return int(x), int(y)


class GeoAwareManager(BaseManager):
    pass

GeoAwareManager.register("GeoAware", GeoAware)


def serve_geo_aware(port=50011):
    manager = GeoAwareManager(address=("localhost", port), authkey=b"geoaware")
    server = manager.get_server()
    print(f"Serving GeoAware on localhost:{port}...")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50011, help="Port to serve GeoAware on")
    args = parser.parse_args()
    serve_geo_aware(args.port)

