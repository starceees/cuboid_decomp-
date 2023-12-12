from __future__ import absolute_import, division, print_function

# Add the path to the LiteMono folder
import sys
sys.path.insert(0, '/Users/tangxinran/Documents/NYU/robot_perception/project/LiteMono')

import os
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

# Assuming these are in the sys.path already
from networks.depth_encoder import LiteMono
from networks.depth_decoder import DepthDecoder
from layers import disp_to_depth

class DepthModel:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config['no_cuda'] else "cpu")
        self.load_model(config)

    def load_model(self, config):
        assert os.path.exists(config['load_weights_folder']), \
            "The weights folder does not exist: {}".format(config['load_weights_folder'])

        print("-> Loading model from ", config['load_weights_folder'])
        encoder_path = os.path.join(config['load_weights_folder'], "encoder.pth")
        decoder_path = os.path.join(config['load_weights_folder'], "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location=self.device)
        decoder_dict = torch.load(decoder_path, map_location=self.device)

        self.feed_height = encoder_dict['height']
        self.feed_width = encoder_dict['width']

        print("   Loading pretrained encoder")
        self.encoder = LiteMono(model=config['model'],
                                height=self.feed_height,
                                width=self.feed_width)
        self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in self.encoder.state_dict()})
        self.encoder.to(self.device)
        self.encoder.eval()

        print("   Loading pretrained decoder")
        self.depth_decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(3))
        self.depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in self.depth_decoder.state_dict()})
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def process_image(self, rgb_image):
        original_width, original_height = rgb_image.size
        rgb_image = rgb_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
        rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(rgb_image)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

        return depth.cpu().numpy()



# # Create an instance of the DepthModel
# depth_model = DepthModel(config)

# Example usage:
# rgb_image = pil.open('path_to_image.jpg').convert('RGB')
# depth_info = depth_model.process_image(rgb_image)
# print(depth_info)
