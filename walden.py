import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_walden(image):
    image = image.astype(np.float32) / 255.0

    image = np.clip(image * 1.1, 0, 1)

    contrast = 0.9
    image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, 0)

    image[:, :, 1] = np.clip(image[:, :, 1] * 1.1, 0, 1)  # Boost green
    image[:, :, 2] = np.clip(image[:, :, 2] * 1.2, 0, 1)  # Boost blue

    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, 400)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, 400)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / np.max(kernel)

    vignette = np.dstack((mask, mask, mask))

    image = image * (0.85 + 0.15 * vignette)

    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image

# Load your image
image_path = 'C:\\Users\\ThinkPad E560\\Desktop\\Instagram filters project\\walden.jpg'
image = cv2.imread(image_path)

walden_image = apply_walden(image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(walden_image, cv2.COLOR_BGR2RGB))
plt.title('Walden Filter')
plt.axis('off')

plt.show()
