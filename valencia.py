import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_valencia(image):
    image = image.astype(np.float32) / 255.0

    image[:, :, 0] = np.minimum(image[:, :, 0] * 1.9, 1.3)
    image[:, :, 1] = np.minimum(image[:, :, 1] * 1.08, 1.0)
    image[:, :, 2] = np.minimum(image[:, :, 2] * 1.15, 1.0)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)
    l = np.clip(l * 1.1, 0, 1)
    lab_image = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)

    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, 1200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, 1200)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / np.max(kernel)

    vignette = np.dstack((mask, mask, mask))

    image = image * (0.95 + 0.05 * vignette)

    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image

image_path = 'C:\\Users\\ThinkPad E560\\Desktop\\Instagram filters project\\valencia.jpg'
image = cv2.imread(image_path)

filtered_image = apply_valencia(image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Valencia Filter')
plt.axis('off')

plt.show()
