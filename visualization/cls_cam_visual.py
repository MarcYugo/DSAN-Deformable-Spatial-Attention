import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse,os
import cv2
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str)
parser.add_argument('--images_dir', type=str, default='images')
parser.add_argument('--cams_dir', type=str, default='cams')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mods = {
    'dsan_t': dsan_t,
    'dsan_s': dsan_s
}

# Grad-CAM函数
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None
        
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        self.handle = self.target_layer.register_forward_hook(forward_hook)
        self.handle = self.target_layer.register_full_backward_hook(backward_hook)
        
    def __call__(self, x):
        self.model.zero_grad()
        return self.model(x)
    
    def get_gradient(self):
        return self.gradient
    
    def get_feature_maps(self):
        return self.feature_maps
    
    def remove_hooks(self):
        self.handle.remove()

# model
model = mods[args.model_name](pretrained=True)
model.to(device)
model.eval()

# transform
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_paths = [f'{args.images_dir}/{p}' for p in os.listdir(args.images_dir)]

for p in img_paths:
    img = Image.open(p)
    try:
        img_tensor = preprocess(img)
    except:
        print(p)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    src_img = cv2.imread(p)
    src_img = cv2.resize(src_img, (256,256))

    # Grad-CAM
    grad_cam = GradCam(model, model.block4[-1])
    output = grad_cam(img_tensor)
    cls_idx = torch.argmax(output).item()    # get prediction class id
    score = output[:, cls_idx].sum()    # get score
    score.backward(retain_graph=True)    # backward

    feature_maps = grad_cam.get_feature_maps()[0]
    gradient = grad_cam.get_gradient()[0]

    weights = torch.mean(gradient, axis=(1, 2), keepdim=True)
    cam = torch.sum(weights * feature_maps, axis=0)
    cam = nn.functional.relu(cam)
    cam_resize = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0),[256,256],mode='bilinear')
    cam_resize = cam_resize[0,0].cpu().detach().numpy()
    cam_resize = (cam_resize - cam_resize.min())/(cam_resize.max() - cam_resize.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resize), cv2.COLORMAP_JET)
    visual_pic = src_img*0.5 + heatmap*0.5
    # save
    id = p.replace(f'{args.images_dir}/ILSVRC2012_val_', '').replace('.JPEG', '')
    cam_path = f'{args.cams_dir}/{id}_{args.model_name}.jpg'
    cv2.imwrite(cam_path, visual_pic)
    print(cam_path)