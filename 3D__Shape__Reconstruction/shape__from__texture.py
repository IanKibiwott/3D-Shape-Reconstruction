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

def shape_from_texture_interactive(image_path):
    """Interactive version with sliders"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:     
        print("Error loading image")
        return
    
    # Create window and trackbars
    window_name = 'Shape from Texture - Interactive'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 500)
    
    # Create trackbars for parameters
    cv2.createTrackbar('Sobel Size', window_name, 5, 15, lambda x: None)
    cv2.createTrackbar('Blur Size', window_name, 15, 30, lambda x: None)
    cv2.createTrackbar('Contrast', window_name, 100, 200, lambda x: None)
    
    print("Shape from Texture - Interactive Controls")
    print("Sobel Size: Sobel kernel size (3-15, odd numbers)")
    print("Blur Size: Gaussian blur kernel size")
    print("Contrast: Depth map contrast (1.0-2.0x)")
    print("Press 'q' to quit, 'r' to reset")
    
    while True:
        # Get trackbar values
        sobel_size = cv2.getTrackbarPos('Sobel Size', window_name)
        blur_size = cv2.getTrackbarPos('Blur Size', window_name)
        contrast = cv2.getTrackbarPos('Contrast', window_name) / 100.0
        
        # Ensure odd kernel sizes
        if sobel_size % 2 == 0:
            sobel_size += 1
        if sobel_size < 3:
            sobel_size = 3
            
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size < 1:
            blur_size = 1
        
        # Compute local texture using gradients with current parameters
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=sobel_size)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=sobel_size)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Apply contrast
        magnitude = magnitude * contrast
        
        # Texture gradient indicates surface tilt
        texture_depth = cv2.GaussianBlur(magnitude, (blur_size, blur_size), 0)
        
        # Normalize for display
        texture_display = cv2.normalize(texture_depth, None, 0, 255, cv2.NORM_MINMAX)
        texture_display = texture_display.astype(np.uint8)
        
        # Combine input and depth map
        combined = np.hstack([img, texture_display])
        
        # Add labels and parameters
        cv2.putText(combined, "Input Texture", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Depth Map", (img.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined, f"Sobel: {sobel_size}", (10, combined.shape[0]-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Blur: {blur_size}", (10, combined.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Contrast: {contrast:.1f}", (10, combined.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.setTrackbarPos('Sobel Size', window_name, 5)
            cv2.setTrackbarPos('Blur Size', window_name, 15)
            cv2.setTrackbarPos('Contrast', window_name, 100)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use interactive version for demo
    shape_from_texture_interactive(r"c:\Users\User\Downloads\IanK25photo.jpeg")
    
    # Or use original version
    # shape_from_texture(r"c:\Users\User\Downloads\IanK25photo.jpeg")python shape__from__texture.py