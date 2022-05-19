import os
import sys
sys.path.insert(0, "stylegan-encoder")
import tempfile
import warnings
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torchvision.transforms as transforms
import dlib
from cog import BasePredictor, Path, Input

from demo import load_checkpoints
from demo import make_animation
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


warnings.filterwarnings("ignore")


PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LANDMARKS_DETECTOR = LandmarksDetector("shape_predictor_68_face_landmarks.dat")


class Predictor(BasePredictor):
    def setup(self):

        self.device = torch.device("cuda:0")
        datasets = ["vox", "taichi", "ted", "mgif"]
        (
            self.inpainting,
            self.kp_detector,
            self.dense_motion_network,
            self.avd_network,
        ) = ({}, {}, {}, {})
        for d in datasets:
            (
                self.inpainting[d],
                self.kp_detector[d],
                self.dense_motion_network[d],
                self.avd_network[d],
            ) = load_checkpoints(
                config_path=f"config/{d}-384.yaml"
                if d == "ted"
                else f"config/{d}-256.yaml",
                checkpoint_path=f"checkpoints/{d}.pth.tar",
                device=self.device,
            )

    def predict(
        self,
        source_image: Path = Input(
            description="Input source image.",
        ),
        driving_video: Path = Input(
            description="Choose a micromotion.",
        ),
        dataset_name: str = Input(
            choices=["vox", "taichi", "ted", "mgif"],
            default="vox",
            description="Choose a dataset.",
        ),
    ) -> Path:

        predict_mode = "relative"  # ['standard', 'relative', 'avd']
        # find_best_frame = False

        pixel = 384 if dataset_name == "ted" else 256

        if dataset_name == "vox":
            # first run face alignment
            align_image(str(source_image), 'aligned.png')
            source_image = imageio.imread('aligned.png')
        else:
            source_image = imageio.imread(str(source_image))
        reader = imageio.get_reader(str(driving_video))
        fps = reader.get_meta_data()["fps"]
        source_image = resize(source_image, (pixel, pixel))[..., :3]

        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [
            resize(frame, (pixel, pixel))[..., :3] for frame in driving_video
        ]

        inpainting, kp_detector, dense_motion_network, avd_network = (
            self.inpainting[dataset_name],
            self.kp_detector[dataset_name],
            self.dense_motion_network[dataset_name],
            self.avd_network[dataset_name],
        )

        predictions = make_animation(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device="cuda:0",
            mode=predict_mode,
        )

        # save resulting video
        out_path = Path(tempfile.mkdtemp()) / "output.mp4"
        imageio.mimsave(
            str(out_path), [img_as_ubyte(frame) for frame in predictions], fps=fps
        )
        return out_path


def align_image(raw_img_path, aligned_face_path):
    for i, face_landmarks in enumerate(LANDMARKS_DETECTOR.get_landmarks(raw_img_path), start=1):
        image_align(raw_img_path, aligned_face_path, face_landmarks)
