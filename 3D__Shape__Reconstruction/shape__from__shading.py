# shape__from__shading.py
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def shape_from_shading_interactive(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:     
        print("Error loading image")
        return
    
    img_original = img.copy()
    img = img / 255.00
    h, w = img.shape
    
    # Create window and trackbars
    window_name = 'Shape from Shading - Interactive'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Make window resizable
    cv2.resizeWindow(window_name, 1000, 500)  # Set initial size
    
    # Create trackbars
    cv2.createTrackbar('Light X', window_name, 50, 100, lambda x: None)
    cv2.createTrackbar('Light Y', window_name, 50, 100, lambda x: None)
    cv2.createTrackbar('Light Z', window_name, 100, 100, lambda x: None)
    cv2.createTrackbar('Alpha', window_name, 50, 100, lambda x: None)
    
    print("Interactive Controls Active!")
    print("If window is too large/small, you can resize it manually")
    
    while True:
        # Get trackbar values
        Lx = (cv2.getTrackbarPos('Light X', window_name) - 50) / 50.0
        Ly = (cv2.getTrackbarPos('Light Y', window_name) - 50) / 50.0
        Lz = cv2.getTrackbarPos('Light Z', window_name) / 100.0
        alpha = cv2.getTrackbarPos('Alpha', window_name) / 100.0
        
        if Lz < 0.1:
            Lz = 0.1
        
        # Reconstruct depth
        z = np.zeros((h, w))
        for i in range(1, h-1):
            for j in range(1, w-1):
                I = img[i, j]
                z[i, j] = np.sqrt(max(0, 1 - (I / Lz) ** 2)) * 255 * alpha
        
        # Convert to display format
        z_display = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)
        z_display = z_display.astype(np.uint8)
        
        # Resize if images are too large for display
        max_display_width = 800
        if w > max_display_width:
            scale = max_display_width / (w * 2)  # Because we have 2 images side by side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_display = cv2.resize(img_original, (new_w, new_h))
            z_display_resized = cv2.resize(z_display, (new_w, new_h))
            combined = np.hstack([img_display, z_display_resized])
        else:
            combined = np.hstack([img_original, z_display])
        
        # Add text
        cv2.putText(combined, f"Light: ({Lx:.1f}, {Ly:.1f}, {Lz:.1f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"Alpha: {alpha:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Original", (10, combined.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "Depth Map", (combined.shape[1]//2 + 10, combined.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.setTrackbarPos('Light X', window_name, 50)
            cv2.setTrackbarPos('Light Y', window_name, 50)
            cv2.setTrackbarPos('Light Z', window_name, 100)
            cv2.setTrackbarPos('Alpha', window_name, 50)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    shape_from_shading_interactive(r"c:\Users\User\Downloads\IanK25photo.jpeg")