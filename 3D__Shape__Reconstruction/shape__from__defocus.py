# shape__from__defocus.py
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def shape_from_defocus(image1_path, image2_path):    
    # Read both grayscale images 
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    # Check if images loaded correctly    
    if img1 is None or img2 is None:     
        print("Error loading images")
        return
    
    # Resize second image to match first image
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Compute laplacian blur and focus difference
    blur1 = cv2.Laplacian(img1, cv2.CV_64F)
    blur2 = cv2.Laplacian(img2, cv2.CV_64F)
    focus_measure = np.abs(blur1) - np.abs(blur2)
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

def shape_from_defocus_interactive(image1_path, image2_path):
    """Interactive version with sliders"""
    # Read both grayscale images 
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:     
        print("Error loading images")
        return
    
    # Resize second image to match first image
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Create window and trackbars
    window_name = 'Shape from Defocus - Interactive'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 400)
    
    # Create trackbars for parameters
    cv2.createTrackbar('Blur Size', window_name, 1, 10, lambda x: None)
    cv2.createTrackbar('Contrast', window_name, 100, 200, lambda x: None)
    cv2.createTrackbar('Scale', window_name, 100, 200, lambda x: None)
    
    print("Shape from Defocus - Interactive Controls")
    print("Blur Size: Kernel size for additional blur")
    print("Contrast: Depth map contrast enhancement")
    print("Scale: Depth map scaling factor")
    print("Press 'q' to quit")
    
    while True:
        # Get trackbar values
        blur_size = cv2.getTrackbarPos('Blur Size', window_name)
        contrast = cv2.getTrackbarPos('Contrast', window_name) / 100.0
        scale = cv2.getTrackbarPos('Scale', window_name) / 100.0
        
        # Apply additional blur if needed
        if blur_size > 0:
            kernel_size = blur_size * 2 + 1
            img1_blur = cv2.GaussianBlur(img1, (kernel_size, kernel_size), 0)
            img2_blur = cv2.GaussianBlur(img2, (kernel_size, kernel_size), 0)
        else:
            img1_blur = img1.copy()
            img2_blur = img2.copy()
        
        # Compute laplacian blur and focus difference
        blur1 = cv2.Laplacian(img1_blur, cv2.CV_64F)
        blur2 = cv2.Laplacian(img2_blur, cv2.CV_64F)
        focus_measure = np.abs(blur1) - np.abs(blur2)
        
        # Apply contrast and scale
        depth_map = focus_measure * contrast * scale
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
        
        # Create display - resize if too large
        max_height = 400
        if img1.shape[0] > max_height:
            scale_factor = max_height / img1.shape[0]
            new_w = int(img1.shape[1] * scale_factor)
            new_h = max_height
            img1_display = cv2.resize(img1, (new_w, new_h))
            img2_display = cv2.resize(img2, (new_w, new_h))
            depth_display = cv2.resize(depth_map, (new_w, new_h))
        else:
            img1_display = img1
            img2_display = img2
            depth_display = depth_map
        
        # Combine all three images
        combined = np.hstack([img1_display, img2_display, depth_display])
        
        # Add labels and parameters
        cv2.putText(combined, "Focus 1", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Focus 2", (img1_display.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Depth Map", (img1_display.shape[1]*2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(combined, f"Blur: {blur_size}", (10, combined.shape[0]-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Contrast: {contrast:.1f}", (10, combined.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Scale: {scale:.1f}", (10, combined.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use interactive version for demo
    #shape_from_defocus_interactive(   
       # r"C:\Users\User\Downloads\IanK25photo.jpeg",  # defocus1
        #r"C:\Users\User\Downloads\334289821-Baltimore_Oriole-Matthew_Plante.jpg"  # defocus2
    #)
    
    # Or use original version
     shape_from_defocus(   
       r"C:\Users\User\Downloads\IanK25photo.jpeg",
       r"C:\Users\User\Downloads\334289821-Baltimore_Oriole-Matthew_Plante.jpg"
     )