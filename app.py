import os
#os.system("pip freeze")
from huggingface_hub import hf_hub_download
#os.system("pip -qq install facenet_pytorch")
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch, PIL
from tqdm.notebook import tqdm
import gradio as gr
import torch

modelarcanev4 = hf_hub_download(repo_id="akhaliq/ArcaneGANv0.4", filename="ArcaneGANv0.4.jit")
modelarcanev3 = hf_hub_download(repo_id="akhaliq/ArcaneGANv0.3", filename="ArcaneGANv0.3.jit")
modelarcanev2 = hf_hub_download(repo_id="akhaliq/ArcaneGANv0.2", filename="ArcaneGANv0.2.jit")


mtcnn = MTCNN(image_size=256, margin=80)

# simplest ye olde trustworthy MTCNN for face detection with landmarks
def detect(img):
 
        # Detect faces
        batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
        # Select faces
        if not mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
            )
 
        return batch_boxes, batch_points

# my version of isOdd, should make a separate repo for it :D
def makeEven(_x):
  return _x if (_x % 2 == 0) else _x+1

# the actual scaler function
def scale(boxes, _img, max_res=1_500_000, target_face=256, fixed_ratio=0, max_upscale=2, VERBOSE=False):
 
    x, y = _img.size
 
    ratio = 2 #initial ratio
 
    #scale to desired face size
    if (boxes is not None):
      if len(boxes)>0:
        ratio = target_face/max(boxes[0][2:]-boxes[0][:2]); 
        ratio = min(ratio, max_upscale)
        if VERBOSE: print('up by', ratio)

    if fixed_ratio>0:
      if VERBOSE: print('fixed ratio')
      ratio = fixed_ratio
 
    x*=ratio
    y*=ratio
 
    #downscale to fit into max res 
    res = x*y
    if res > max_res:
      ratio = pow(res/max_res,1/2); 
      if VERBOSE: print(ratio)
      x=int(x/ratio)
      y=int(y/ratio)
 
    #make dimensions even, because usually NNs fail on uneven dimensions due skip connection size mismatch
    x = makeEven(int(x))
    y = makeEven(int(y))
    
    size = (x, y)

    return _img.resize(size)

""" 
    A useful scaler algorithm, based on face detection.
    Takes PIL.Image, returns a uniformly scaled PIL.Image
    boxes: a list of detected bboxes
    _img: PIL.Image
    max_res: maximum pixel area to fit into. Use to stay below the VRAM limits of your GPU.
    target_face: desired face size. Upscale or downscale the whole image to fit the detected face into that dimension.
    fixed_ratio: fixed scale. Ignores the face size, but doesn't ignore the max_res limit.
    max_upscale: maximum upscale ratio. Prevents from scaling images with tiny faces to a blurry mess.
"""

def scale_by_face_size(_img, max_res=1_500_000, target_face=256, fix_ratio=0, max_upscale=2, VERBOSE=False):
    boxes = None
    boxes, _ = detect(_img)
    if VERBOSE: print('boxes',boxes)
    img_resized = scale(boxes, _img, max_res, target_face, fix_ratio, max_upscale, VERBOSE)
    return img_resized


size = 256

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

t_stds = torch.tensor(stds).cuda().half()[:,None,None]
t_means = torch.tensor(means).cuda().half()[:,None,None]

def makeEven(_x):
  return int(_x) if (_x % 2 == 0) else int(_x+1)

img_transforms = transforms.Compose([                        
            transforms.ToTensor(),
            transforms.Normalize(means,stds)])
 
def tensor2im(var):
     return var.mul(t_stds).add(t_means).mul(255.).clamp(0,255).permute(1,2,0)

def proc_pil_img(input_image, model):
    transformed_image = img_transforms(input_image)[None,...].cuda().half()
            
    with torch.no_grad():
        result_image = model(transformed_image)[0]
        output_image = tensor2im(result_image)
        output_image = output_image.detach().cpu().numpy().astype('uint8')
        output_image = PIL.Image.fromarray(output_image)
    return output_image
    
modelv4 = torch.jit.load(modelarcanev4).eval().cuda().half()
modelv3 = torch.jit.load(modelarcanev3).eval().cuda().half()
modelv2 = torch.jit.load(modelarcanev2).eval().cuda().half()

def process(im, version):
    if version == 'version 0.4':
        #im = scale_by_face_size(im, target_face=256, max_res=1_500_000, max_upscale=1)
        im = scale_by_face_size(im, target_face=256, max_res=2_2048_000, max_upscale=4)
        res = proc_pil_img(im, modelv4)
    elif version == 'version 0.3':
        im = scale_by_face_size(im, target_face=256, max_res=1_500_000, max_upscale=1)
        res = proc_pil_img(im, modelv3)
    else:
        im = scale_by_face_size(im, target_face=256, max_res=1_500_000, max_upscale=1)
        res = proc_pil_img(im, modelv2)
    return res
        
title = "ArcaneGAN"
description = "Gradio demo for ArcaneGAN, portrait to Arcane style. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<div style='text-align: center;'>ArcaneGan by <a href='https://github.com/yesidx08x' target='_blank'>Alexander S</a> | <a href='https://github.com/yesidx08x/ArcaneGAN' target='_blank'>Github Repo</a> | <center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_arcanegan' alt='visitor badge'></center></div>"

gr.Interface(
    process, 
    [gr.inputs.Image(type="pil", label="Input"),gr.inputs.Radio(choices=['version 0.2','version 0.3','version 0.4'], type="value", default='version 0.4', label='version')
], 
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[['bill.png','version 0.4'],['keanu.png','version 0.4'],['will.jpeg','version 0.4']],
    allow_flagging=False,
    allow_screenshot=True
    ).launch(enable_queue=True,cache_examples=True,share=True)
