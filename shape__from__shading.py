# shape__from__shading.py
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
def shape_from_shading(image_path, light_dir=(0, 0, 1), alpha= 0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:     
        print("Error loading images")
        return
    img = img / 255.00
    h, w = img.shape
    p= np.zeros((h, w))
    q= np.zeros((h, w))
    z= np.zeros((h, w))
    Lx, Ly, Lz = light_dir
    for i in range (1, h-1):
        for j in range(1, w - 1):
                I = img[i, j]
                p[i, j] = alpha * (I * Lx)
                q[i, j] = alpha * (I * Ly)
                z[i, j] = np.sqrt(max(0, 1  -(I / Lz) **2)) * 255

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Shading Image")
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Depth(Shape from Shading)")
    plt.imshow(z, cmap='gray')
    plt.show()

if __name__ == "__main__":
    shape_from_shading(r"c:\Users\User\Downloads\IanK25photo.jpeg")
