import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image
import cv2
import torchvision
import argparse
from typing import List
import keras
import tensorflow as tf
from math import log10, sqrt 
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def parse_arg():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str,
                        default=os.path.join('Test', 'john.jpg'))
    parser.add_argument("--gpu", type=bool, default=False)
    args = parser.parse_args()
    return args
    
image_path = 'Test/john.jpg'
out_path = 'Test/john.jpg'


class UnionFind:

    def __init__(self, n_vertices):
        """Initialize."""
        self.parent = [*range(n_vertices)]

    def find(self, id_v):
        """Find current root of a vertex whose index is id_v."""
        while self.parent[id_v] != id_v:
            id_v = self.parent[id_v]
        return id_v

    def union(self, id1, id2):
        """Unite two vertives whose indices are id1 and id2.

        @returns (int, int) root of vertex id1, root of vertex id2
                            before union.
        """
        root2 = self.find(id2)
        root1 = self.find(id1)
        self.parent[root1] = root2
        return root1, root2


def load_model(download=True):
    """Load pretrained Mask R-CNN model with a Resnet50 as backbone."""
    print("Loading model...")
    if download:
        print("Downloading model...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=download, pretrained_backbone=download
    )
    if not download:
        model.load_state_dict(torch.load(os.path.join(
            'model', 'maskrcnn_resnet50_fpn.ptn')))
    else:
        torch.save(model.state_dict(), os.path.join(
            'model', 'maskrcnn_resnet50_fpn.ptn'))
            

    print("Done!")
    return model


def IoU(prediction, id1, id2):
    """Compute IoU of two bounding boxes whose indices are id1 and id2."""
    # First extract the bounding boxes for each mask.
    bbox1 = prediction['boxes'][id1]
    bbox2 = prediction['boxes'][id2]
    device = bbox1.device

    # Compute U.
    xmin = min(bbox1[0], bbox2[0])
    ymin = min(bbox1[1], bbox2[1])
    xmax = max(bbox1[2], bbox2[2])
    ymax = max(bbox1[3], bbox2[3])
    u = (xmax - xmin) * (ymax - ymin)
    union_bbox = torch.FloatTensor([xmin, ymin, xmax, ymax]).to(device)

    # Compute I.
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    i = (xmax - xmin) * (ymax - ymin)
    inter_bbox = torch.FloatTensor([xmin, ymin, xmax, ymax]).to(device)

    return i/u, inter_bbox, union_bbox


def IoU_mask(prediction, id1, id2, thres):
    """Compute IoU  of 2 masks."""
    m1 = prediction['masks'][id1, 0]
    m2 = prediction['masks'][id2, 0]
    inter = (m1 > thres) & (m2 > thres)
    union = (m1 > thres) | (m2 > thres)

    return torch.sum(inter).float() / torch.sum(union)


def merge_masks(m1, m2):
    """Merge masks."""
    return torch.max(m1, m2)


def get_mask(prediction,
             thres_merge_per=0, thres_merge_obj=0.01,
             thres=0.1, thres_score=0.7):
    """Merge or select masks for the bokeh effect."""
    num_objs = prediction['labels'].shape[0]
    # print(prediction['labels'] == 1)

    # Merge mask of crowds
    idx_person = torch.arange(num_objs)[torch.logical_and(
        prediction['labels'] == 1, prediction['scores'] > thres_score)]
    # print(idx_person)

    uf = UnionFind(len(idx_person))

    for j in range(len(idx_person)):
        for i in range(j+1, len(idx_person)):
            iou, inter, union = IoU(prediction, idx_person[i], idx_person[j])
            if iou > thres_merge_per:
                # Then make edge, unite and merge two masks
                root_i, root_j = uf.union(i, j)
                prediction['masks'][idx_person[root_j], 0] = merge_masks(
                    prediction['masks'][idx_person[root_i], 0],
                    prediction['masks'][idx_person[root_j], 0]
                )
    # Select main group of person-masks
    max_area = 0
    main_idx = -1
    # Center region: (x_min, y_min, x_max, y_max)
    h, w = prediction['masks'][0, 0].shape
    x_min = w // 6
    x_max = w - w // 6
    y_min = h // 6
    y_max = h - h // 6
    for i in range(len(idx_person)):
        root = uf.find(i)
        area_i = torch.sum(
            prediction['masks'][idx_person[root], 0, y_min:y_max, x_min:x_max]
            > 0
        )
        if area_i > max_area:
            max_area = area_i
            main_idx = idx_person[root]

    # Merge masks of objects overlapping on main persons
    idx_obj = torch.arange(num_objs)[prediction['labels'] != 1]
    for i in range(len(idx_obj)):
        iou = IoU_mask(prediction, main_idx, idx_obj[i], thres)
        if iou > thres_merge_obj:
            prediction['masks'][main_idx, 0] = merge_masks(
                prediction['masks'][main_idx, 0],
                prediction['masks'][idx_obj[i], 0]
            )

    return prediction['masks'][main_idx, 0]


def disc_shaped_kernel(ksize):
    """Return an average disc-shape kernel."""
    k = ksize // 2
    x, y = np.ogrid[-k:k+1, -k:k+1]
    mask = x**2 + y**2 <= k**2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[mask] = 1.0
    kernel = cv2.blur(kernel, (3, 3))
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_blur(image, prediction, thres, degree=1/30, gamma=0.25):
    """Synthesize image with bokeh effect.

    @param degree (int) [0, 1]
           The larger it is, the more blurred the background
    """
    h, w = image.shape[0:2]
    ksize = int(min(h, w) * degree)
    if ksize % 2 == 0:
        ksize += 1

    mask = get_mask(prediction, thres=thres).detach().cpu().numpy()
    mask[mask > thres] = 1.0
    mask[mask < thres] = 0.0
    mask = cv2.erode(mask, np.ones((ksize//4, ksize//4), dtype=np.uint8))
    closing = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((ksize, ksize), dtype=np.uint8))
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask_dilated = cv2.dilate(mask, np.ones((ksize, ksize),
                                            dtype=np.uint8)
                              )
    mask_dilated = cv2.GaussianBlur(mask_dilated, (ksize, ksize), 0)
    mask_dilated = np.expand_dims(mask_dilated, 2)

    # Bokeh effect on the whole image
    kernel = disc_shaped_kernel(ksize)

    # Get image gamma-corrected and apply disc-shape blur to get an
    # image with bokeh effect
    gamma_correction = ((image / 255.0) ** (1 / gamma)) * 255
    bokeh = cv2.filter2D(gamma_correction.astype(np.uint8), 3, kernel)
    bokeh = (((bokeh / 255.0) ** gamma) * 255).astype(np.uint8)
    # Keep only pixels with high value from bokeh image
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    bokeh = bokeh * (1.0 - mask_dilated)
    bokeh = cv2.max(bokeh.astype(np.uint8), blurred)

    # Blend
    mask = np.expand_dims(mask, 2)
    image = image * mask + bokeh * (1 - mask)
    return image.astype(np.uint8)


#if __name__ == "__main__":
args = parse_arg()
args = vars(args)

if args['gpu']:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("CUDA is not available, switched to CPU!")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
print(device)

if not os.path.exists('model'):
    os.makedirs('model')

if len(os.listdir('model')) > 0:
    model = load_model(download=False).to(device)
else:
    model = load_model(download=True).to(device)
model.eval()

if not os.path.exists('output'):
    os.makedirs('output')

if os.path.isfile(args['input_path']):
    img = Image.open(args['input_path'])
    img_np = np.array(img, dtype=np.uint8)[:, :, ::-1]
    cv2.imshow("Original image", img_np)
    cv2.waitKey(0)
    img = torchvision.transforms.functional.to_tensor(img)

    # Predict masks
    with torch.no_grad():
        predictions = model([img.to(device)])

    out = apply_blur(img_np, predictions[0], thres=0.5)
    cv2.imshow("Image with Bokeh effect", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(
        os.path.join('output', os.path.split(args['input_path'])[-1]),
        out
    )

elif os.path.isdir(args['input_path']):
    img_names = sorted(os.listdir(args['input_path']))
    for img_name in img_names:
        print(img_name)
        img = Image.open(
            os.path.join(args['input_path'], img_name)
        ).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)[:, :, ::-1]
        img = torchvision.transforms.functional.to_tensor(img)

        # Predict masks
        with torch.no_grad():
            predictions = model([img.to(device)])

        out = apply_blur(img_np, predictions[0], thres=0.5)
        plt.imsave(os.path.join("{}_blurred.png".format(img_name)), out)
        
        #(os.path.join(output_directory, "{}_blurred.png".format(output_name)), result)
        
        

#PSNR

def PSNR(original, compressed):
 mse = np.mean((original - compressed) ** 2)
 if(mse == 0):
  return 100
 max_pixel = 255.0
 psnr = 20 * log10(max_pixel / sqrt(mse))
 return psnr 
  
# Import images
original = cv2.imread(image_path)
compressed = cv2.imread('Test/john.jpg')

# Check for same size and ratio and report accordingly
ho, wo, _ = original.shape
hc, wc, _ = compressed.shape
ratio_orig = ho/wo
ratio_comp = hc/wc
dim = (wc, hc)

if round(ratio_orig, 2) != round(ratio_comp, 2):
 print("\nImages not of the same dimension. Check input.")
 exit()

# Resize original if the compressed image is smaller
elif ho > hc and wo > wc:
 print("\nResizing original image for analysis...")
 original = cv2.resize(original, dim)

elif ho < hc and wo < wc:
 print("\nCompressed image has a larger dimension than the original. Check input.")
 exit()

value = PSNR(original, compressed)
print("\nPeak Signal-to-Noise Ratio (PSNR) value is", value, "dB")


#SSIM

def mse(imageA, imageB):
 # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
 mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
 mse_error /= float(imageA.shape[0] * imageA.shape[1])
	
 # return the MSE. The lower the error, the more "similar" the two images are.
 return mse_error

def compare(imageA, imageB):
 # Calculate the MSE and SSIM
 m = mse(imageA, imageB)
 s = ssim(imageA, imageB)

 # Return the SSIM. The higher the value, the more "similar" the two images are.
 return s


# Import images
image1 = cv2.imread(image_path)
image2 = cv2.imread('Test/john.jpg')   

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Check for same size and ratio and report accordingly
ho, wo, _ = image1.shape
hc, wc, _ = image2.shape
ratio_orig = ho/wo
ratio_comp = hc/wc
dim = (wc, hc)

if round(ratio_orig, 2) != round(ratio_comp, 2):
 print("\nImages not of the same dimension. Check input.")
 exit()

# Resize first image if the second image is smaller
elif ho > hc and wo > wc:
 #print("\nResizing original image for analysis...")
 gray1 = cv2.resize(gray1, dim)

elif ho < hc and wo < wc:
 print("\nCompressed image has a larger dimension than the original. Check input.")
 exit()

if round(ratio_orig, 2) == round(ratio_comp, 2):
 mse_value = mse(gray1, gray2)
 ssim_value = compare(gray1, gray2)
 print("MSE:", mse_value)
 print("SSIM:", ssim_value)