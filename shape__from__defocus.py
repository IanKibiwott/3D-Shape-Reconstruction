# shape__from__defocus.py
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
def shape_from_defocus(image1_path, image2_path):    
    #Read both grayscale images 
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    #check if images loaded correctly    
    if img1 is None or img2 is None:     
        print("Error loading images")
        return
        # Resize second image to match first image
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    #compute laplacian blur and focus  difference
    blur1 =    cv2.Laplacian(img1, cv2.CV_64F)
    blur2 =    cv2.Laplacian(img2, cv2.CV_64F)
    focus_measure =    np.abs(blur1) -np.abs(blur2)
    depth_map = cv2.normalize(focus_measure, None, 0, 255, cv2.NORM_MINMAX)

    # Display results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Focus 1")
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Focus 2")
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Depth from Defocus")
    plt.imshow(depth_map, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
            
if __name__ == "__main__":
    shape_from_defocus(   
        r"C:\Users\User\Downloads\IanK25photo.jpeg",  # defocus1
        r"C:\Users\User\Downloads\334289821-Baltimore_Oriole-Matthew_Plante.jpg"  # defocus2
    )