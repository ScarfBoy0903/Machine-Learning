import skimage.io
import numpy as np
import sys
import os
from os import listdir
img_folder = sys.argv[1]

image_names = listdir(img_folder)
image = []

for name in image_names:
    image.append(skimage.io.imread(os.path.join(img_folder,name)))
image_flat = np.reshape(image,(415,-1))
mean_face = np.mean(image_flat,axis=0)
U, S, V = np.linalg.svd((image_flat - mean_face).T, full_matrices=False)

input_img = skimage.io.imread(os.path.join(img_folder,sys.argv[2])).flatten()
output = mean_face + np.dot(np.dot(input_img - mean_face, U[:, :4]), U[:, :4].T)
output = output - np.min(output)
output = output / np.max(output)
output = (output * 255).astype(np.uint8)

skimage.io.imsave("reconstruction.jpg", output.reshape(600,600,3))