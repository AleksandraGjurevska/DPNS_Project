import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_lark(image):
    image = image.astype(np.float32) / 255.0

    brightness = 0.4
    contrast = 1.3
    red_channel = [0.9, 0.0]
    green_channel = [1.5, 0.0]
    blue_channel = [1.3, 0.0]

    image = cv2.addWeighted(image, contrast, image, 0, brightness - 0.5)

    image[:, :, 2] = np.minimum(1.0, np.maximum(0.0, image[:, :, 2] * red_channel[0] + red_channel[1]))
    image[:, :, 1] = np.minimum(1.0, np.maximum(0.0, image[:, :, 1] * green_channel[0] + green_channel[1]))
    image[:, :, 0] = np.minimum(1.0, np.maximum(0.0, image[:, :, 0] * blue_channel[0] + blue_channel[1]))

    image = (image * 255).astype(np.uint8)

    return image

image_path = 'C:\\Users\\ThinkPad E560\\Desktop\\Instagram filters project\\lark.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

filtered_image = apply_lark(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image)
plt.title('Lark Filter Applied')
plt.axis('off')

plt.show()
