import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#image path
current_path = os.path.dirname(os.path.abspath(__file__))
image_path = r'\images\ml.jpg'
image = mpimg.imread(current_path + image_path)

print(image.shape) # (height, width, channels)
print(image[0,0,0]) # Value of the first pixel's red channel
print(image[0,0,:]) # Value of the first pixel in all channels
print(image[0:10,0:10,0]) # Values of the first 10x10 pixels in all channels