# shape__from__texture.py
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
def shape_from_texture(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:     
        print("Error loading images")
        return
    # compute local texture using gradients
    gx = cv2.Sobel(img, cv2.CV_32F , 1, 0,  ksize=5)
    gy = cv2.Sobel(img, cv2.CV_32F , 0, 1,  ksize=5)
    magnitude = np.sqrt(gx**2 + gy**2)

     # Texture gradient indicates surface tilt
    texture_depth = cv2.GaussianBlur(magnitude, (15, 15), 0)
     
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Texture Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Estimated Depth(Shape from Texture)")
    plt.imshow(texture_depth, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
     shape_from_texture(r"c:\Users\User\Downloads\IanK25photo.jpeg")
                    
