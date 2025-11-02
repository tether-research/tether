import torch
from PIL import Image
import numpy as np
from multiprocessing.managers import BaseManager
import hashlib

import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images

from utils.timer_utils import Timer
timer = Timer()

model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
device = "cuda:1"
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)


class Mast3r:
    def __init__(self):
        self.source_image, self.target_image = None, None
        self.matches_im0, self.matches_im1 = None, None
        self.max_cache_size = 10000

    def get_resized(self):
        return self.source_image, self.target_image
    
    def load_images(self, source_path, target_path, cache_path=None, refresh_cache=False):
        cache_filepath = None
        if cache_path is not None:    
            with open(source_path, "rb") as file:
                cache_name = str(hashlib.sha256(file.read()).hexdigest()[:16])
            with open(target_path, "rb") as file:
                cache_name += str(hashlib.sha256(file.read()).hexdigest()[:16])
            cache_filepath = cache_path / f"{cache_name}.npz"

        if cache_filepath is not None and cache_filepath.exists() and not refresh_cache:
            cache_data = np.load(cache_filepath)
            self.matches_im0 = cache_data["im0"]
            self.matches_im1 = cache_data["im1"]
            self.source_size = cache_data["source_size"]
            self.target_size = cache_data["target_size"]
            return True

        timer.start("load_images")
        timer.start("preprocess")
        self.source_image_original = Image.open(source_path).convert('RGB')
        self.target_image_original = Image.open(target_path).convert('RGB')
        images = load_images([source_path, target_path], size=512)
        self.source_image = images[0]
        self.target_image = images[1]
        timer.end("preprocess")

        with torch.no_grad():
            timer.start("get_processed_features_source")
            timer.start("get_processed_features_target")
            output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
            timer.end("get_processed_features_source")
            timer.end("get_processed_features_target")
            timer.start("get_matches")
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']
            desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
            matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=device, dist='dot', block_size=2**13)
            
            H0, W0 = view1['true_shape'][0]
            valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
                matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

            H1, W1 = view2['true_shape'][0]
            print(H0, H1, W0, W1)
            valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
                matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

            valid_matches = valid_matches_im0 & valid_matches_im1
            self.matches_im0, self.matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
            self.source_size = self.source_image_original.size
            self.target_size = self.target_image_original.size
            timer.end("get_matches")

        if cache_filepath is not None:
            data = {"im0": self.matches_im0, "im1": self.matches_im1, "source_size": self.source_image_original.size, "target_size": self.target_image_original.size}
            np.savez(cache_filepath, **data)

        timer.start("delete_cache")
        if cache_path is not None:
            cache_files = list(cache_path.glob("*"))
            if len(cache_files) > self.max_cache_size:
                cache_files.sort(key=lambda f: f.stat().st_atime)
                num_to_delete = len(cache_files) - self.max_cache_size
                for f in cache_files[:num_to_delete]:
                    try:
                        f.unlink()
                    except Exception as e:
                        print(f"Failed to delete {f}: {e}")
        timer.end("delete_cache")
        timer.end("load_images")
        return False

    def get_matches(self):
        return self.matches_im0, self.matches_im1

    def compute_correspondence(self, x, y, max_error):
        print("Computing correspondence...")
        x, y = self.original_to_resized(x, y, self.source_size[0], self.source_size[1])
        dists = np.linalg.norm(self.matches_im0 - np.array([[x, y]]), axis=1)
        closest_idx = np.argmin(dists)
        closest_idx_error = dists[closest_idx]
        if closest_idx_error > max_error:
            print("No correspondence found")
            return None, closest_idx_error
        else:
            print("Correspondence found")
            xy = self.matches_im1[closest_idx]
            x, y = self.resized_to_original(xy[0], xy[1], self.target_size[0], self.target_size[1])
            return (x, y), closest_idx_error
    
    def get_source_info(self):
        return {
            "image_original": self.source_image_original,
            "image": self.source_image_original,
        }
    
    def get_target_info(self):
        return {
            "image_original": self.target_image_original,
            "image": self.target_image_original,
        }

    def original_to_resized(self, x, y, w, h):
        x = x * 512 / w
        y = y * 288 / h
        return int(x), int(y)
    
    def resized_to_original(self, x, y, w, h):
        x = x * w / 512
        y = y * h / 288
        return int(x), int(y)


class Mast3rManager(BaseManager):
    pass

Mast3rManager.register("Mast3r", Mast3r)


def serve_mast3r():
    manager = Mast3rManager(address=("localhost", 50022), authkey=b"mast3r")
    server = manager.get_server()
    print("Serving Mast3r on localhost:50022...")
    server.serve_forever()


if __name__ == "__main__":
    serve_mast3r()
