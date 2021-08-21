#import gpu_imports
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import predict_depth
from deeplab_model import DeepLabModel
from math import log10, sqrt 
import cv2 
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=config)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

image_path = 'D:/thesis/final_models/Model2/portrait-mode-master/test/sherlock.jpg'
pretrained_deeplabv3_path = 'models/deeplabv3_pascal_train_aug'
pretrained_monodepth_path = 'models/model_city2kitti/model_city2kitti'

MAX_INPUT_SIZE = 513

orig_img = cv2.imread(image_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
H, W, C = orig_img.shape
resize_ratio = 1.0 * MAX_INPUT_SIZE / max(H, W)
H, W = int(resize_ratio * H), int(resize_ratio * W)
orig_img = cv2.resize(orig_img, (W, H), cv2.INTER_AREA)

disp_pp = predict_depth.predict(pretrained_monodepth_path, orig_img)
disp_pp = cv2.resize(disp_pp.squeeze(), (W, H))
disp_pp = disp_pp / disp_pp.max()

model = DeepLabModel(pretrained_deeplabv3_path)

seg_map = model.run(orig_img)
obj_mask = seg_map > 0

result = orig_img.copy()
mask_viz = np.ones_like(obj_mask, dtype=np.float32)
threshs = [0.8, 0.5, 0.3]
kernels = [5, 9, 11]
fg_masks = [disp_pp < thresh for thresh in threshs]
for i, fg_mask in enumerate(fg_masks):
    kernel_size = kernels[i]
    blurred = cv2.GaussianBlur(orig_img, (kernel_size, kernel_size), 0)
    result[fg_mask] = blurred[fg_mask]
    mask_viz[fg_mask] = 1.0 - ((i + 1) / len(threshs))
result[obj_mask] = orig_img[obj_mask]
merged_mask = np.max([obj_mask.astype(np.float32),
                      mask_viz], axis=0)

output_directory = os.path.dirname(image_path)
output_name = os.path.splitext(os.path.basename(image_path))[0]

plt.imsave(os.path.join(output_directory, "{}_disp.png".format(
    output_name)), disp_pp, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_fg.png".format(
    output_name)), mask_viz, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_segmap.png".format(
    output_name)), obj_mask, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_mask.png".format(
    output_name)), merged_mask, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_blurred.png".format(output_name)), result)
plt.imsave(os.path.join(output_directory, "{}_resized.png".format(
    output_name)), orig_img)


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
compressed = cv2.imread(os.path.join(output_directory, "{}_blurred.png".format(output_name)))

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
image2 = cv2.imread(os.path.join(output_directory, "{}_blurred.png".format(output_name)))  

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