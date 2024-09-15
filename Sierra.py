import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sierra(image):
    image = image.astype(np.float32) / 255.0

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] *= 0.8
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 1)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    image[:, :, 1] = np.minimum(image[:, :, 1] * 1.1, 1.0)
    image[:, :, 2] = np.minimum(image[:, :, 2] * 1.2, 1.0)

    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, 500)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, 500)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / np.max(kernel)

    vignette = np.dstack((mask, mask, mask))

    image = image * (0.8 + 0.2 * vignette)

    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image

image_path = 'C:\\Users\\ThinkPad E560\\Desktop\\Instagram filters project\\sierra.jpg'
image = cv2.imread(image_path)

filtered_image = apply_sierra(image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Sierra Filter')
plt.axis('off')

plt.show()
