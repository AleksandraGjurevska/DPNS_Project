import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clarendon(image):
    image = image.astype(np.float32) / 255.0

    brightness = 0.1
    contrast = 1.2
    red_channel = [1.0, 0.0]
    green_channel = [1.5, 0.0]
    blue_channel = [1.0, 0.0]

    image = cv2.addWeighted(image, contrast, image, 0, brightness - 0.5)

    image[:, :, 2] = np.minimum(1.0, np.maximum(0.0, image[:, :, 2] * red_channel[0] + red_channel[1]))
    image[:, :, 1] = np.minimum(1.0, np.maximum(0.0, image[:, :, 1] * green_channel[0] + green_channel[1]))
    image[:, :, 0] = np.minimum(1.0, np.maximum(0.0, image[:, :, 0] * blue_channel[0] + blue_channel[1]))

    image = (image * 255).astype(np.uint8)

    return image

image_path = 'C:\\Users\\ThinkPad E560\\Desktop\\Instagram filters project\\claredon.jpg'
image = cv2.imread(image_path)

filtered_image = apply_clarendon(image)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Clarendon Filter')

plt.show()
