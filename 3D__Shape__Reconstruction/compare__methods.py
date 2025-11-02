# compare__methods.py
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import cv2

def create_ground_truth(height=100, width=100, shape_type='sphere'):
    """Create synthetic ground truth 3D shapes"""
    y, x = np.ogrid[-2:2:height*1j, -2:2:width*1j]
    
    if shape_type == 'sphere':
        # Sphere shape
        z = np.sqrt(np.maximum(4 - (x**2 + y**2), 0)) * 40
    elif shape_type == 'plane':
        # Inclined plane
        z = (x + y) * 20 + 50
    elif shape_type == 'cone':
        # Cone shape
        z = np.maximum(3 - np.sqrt(x**2 + y**2), 0) * 30
    elif shape_type == 'ridge':
        # Ridge/wave pattern
        z = (np.sin(3*x) + 1) * 30
    else:
        # Random surface
        z = np.random.rand(height, width) * 50 + 50
    
    return np.clip(z, 0, 255).astype(np.uint8)

def calculate_metrics(predicted, ground_truth):
    """Calculate quantitative metrics"""
    # Ensure same size
    if predicted.shape != ground_truth.shape:
        predicted = cv2.resize(predicted, (ground_truth.shape[1], ground_truth.shape[0]))
    
    # Convert to float for calculations
    pred_float = predicted.astype(np.float64)
    gt_float = ground_truth.astype(np.float64)
    
    # Mean Squared Error
    mse = np.mean((pred_float - gt_float) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_float - gt_float))
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'Mean_Absolute_Error': mae
    }

def quantitative_analysis(shading_depth, texture_depth, defocus_depth, ground_truth):
    """Perform comprehensive quantitative analysis"""
    print("\n" + "="*60)
    print("QUANTITATIVE ACCURACY ANALYSIS")
    print("="*60)
    
    # Calculate metrics for each method
    shading_metrics = calculate_metrics(shading_depth, ground_truth)
    texture_metrics = calculate_metrics(texture_depth, ground_truth) 
    defocus_metrics = calculate_metrics(defocus_depth, ground_truth)
    
    # Print results
    print(f"{'METRIC':<20} {'Shading':<12} {'Texture':<12} {'Defocus':<12} {'BEST'}")
    print("-" * 60)
    
    for metric in ['MSE', 'PSNR', 'Mean_Absolute_Error']:
        values = [
            shading_metrics[metric],
            texture_metrics[metric], 
            defocus_metrics[metric]
        ]
        methods = ['Shading', 'Texture', 'Defocus']
        
        if metric == 'MSE' or metric == 'Mean_Absolute_Error':
            # Lower is better
            best_idx = np.argmin(values)
            best_method = methods[best_idx]
        else:
            # Higher is better (PSNR)
            best_idx = np.argmax(values)
            best_method = methods[best_idx]
        
        if metric == 'PSNR':
            print(f"{metric:<20} {values[0]:<12.2f} {values[1]:<12.2f} {values[2]:<12.2f} {best_method}")
        else:
            print(f"{metric:<20} {values[0]:<12.4f} {values[1]:<12.4f} {values[2]:<12.4f} {best_method}")
    
    print("="*60)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Ground truth and methods
    im0 = axes[0, 0].imshow(ground_truth, cmap='viridis')
    axes[0, 0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(shading_depth, cmap='viridis')
    axes[0, 1].set_title(f'Shape from Shading\nMSE: {shading_metrics["MSE"]:.2f}')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(texture_depth, cmap='viridis')
    axes[0, 2].set_title(f'Shape from Texture\nMSE: {texture_metrics["MSE"]:.2f}')
    plt.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[1, 0].imshow(defocus_depth, cmap='viridis')
    axes[1, 0].set_title(f'Shape from Defocus\nMSE: {defocus_metrics["MSE"]:.2f}')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Error maps
    error_shading = np.abs(shading_depth.astype(float) - ground_truth.astype(float))
    axes[1, 1].imshow(error_shading, cmap='hot')
    axes[1, 1].set_title(f'Shading Error\nMAE: {shading_metrics["Mean_Absolute_Error"]:.2f}')
    plt.colorbar(im0, ax=axes[1, 1])
    
    error_texture = np.abs(texture_depth.astype(float) - ground_truth.astype(float))
    axes[1, 2].imshow(error_texture, cmap='hot')
    axes[1, 2].set_title(f'Texture Error\nMAE: {texture_metrics["Mean_Absolute_Error"]:.2f}')
    plt.colorbar(im0, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    return shading_metrics, texture_metrics, defocus_metrics

def compare_results(shading_depth, texture_depth, defocus_depth, ground_truth=None):
    """Enhanced comparison with quantitative analysis"""
    
    if ground_truth is not None:
        # Run quantitative analysis
        quantitative_analysis(shading_depth, texture_depth, defocus_depth, ground_truth)
    
    # Original 3D visualization
    fig = plt.figure(figsize=(15, 5))
    
    if ground_truth is not None:
        ax0 = fig.add_subplot(141, projection='3d')
        X, Y = np.meshgrid(range(ground_truth.shape[1]), range(ground_truth.shape[0]))
        ax0.plot_surface(X, Y, ground_truth, cmap='viridis', alpha=0.8)
        ax0.set_title("Ground Truth")
        plots = [fig.add_subplot(142, projection='3d'), 
                fig.add_subplot(143, projection='3d'),
                fig.add_subplot(144, projection='3d')]
    else:
        plots = [fig.add_subplot(131, projection='3d'),
                fig.add_subplot(132, projection='3d'),
                fig.add_subplot(133, projection='3d')]
    
    X, Y = np.meshgrid(range(shading_depth.shape[1]), range(shading_depth.shape[0]))
    
    plots[0].plot_surface(X, Y, shading_depth, cmap='plasma', alpha=0.8)
    plots[0].set_title("Shape from Shading")
    
    plots[1].plot_surface(X, Y, texture_depth, cmap='plasma', alpha=0.8)
    plots[1].set_title("Shape from Texture")
    
    plots[2].plot_surface(X, Y, defocus_depth, cmap='plasma', alpha=0.8)
    plots[2].set_title("Shape from Defocus")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create synthetic ground truth
    print("Generating synthetic ground truth...")
    ground_truth = create_ground_truth(100, 100, 'sphere')
    
    # Simulate reconstruction results (simplified - use your actual methods)
    print("Simulating reconstruction results...")
    
    # Add some realistic variations to simulate different method performances
    shading_reconstructed = ground_truth.astype(float) * 0.9 + np.random.normal(0, 8, ground_truth.shape)
    texture_reconstructed = ground_truth.astype(float) * 0.85 + np.random.normal(0, 12, ground_truth.shape)
    defocus_reconstructed = ground_truth.astype(float) * 0.95 + np.random.normal(0, 6, ground_truth.shape)
    
    shading_reconstructed = np.clip(shading_reconstructed, 0, 255).astype(np.uint8)
    texture_reconstructed = np.clip(texture_reconstructed, 0, 255).astype(np.uint8)
    defocus_reconstructed = np.clip(defocus_reconstructed, 0, 255).astype(np.uint8)
    
    # Run comprehensive comparison
    compare_results(shading_reconstructed, texture_reconstructed, defocus_reconstructed, ground_truth)