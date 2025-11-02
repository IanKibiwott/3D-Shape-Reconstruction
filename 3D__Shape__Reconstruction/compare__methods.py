#compare__methods.py
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def compare_results(shading_depth, texture_depth, defocus_depth):
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    #create meshgrid properly (needs both X and Y ranges)
    X ,Y = np.meshgrid(range(shading_depth.shape[1]), range(shading_depth.shape[0]))
    ax1.plot_surface(X, Y, shading_depth, cmap='gray')
    ax1.set_title("Shape from Shading")

    ax2.plot_surface(X, Y, texture_depth, cmap='gray')
    ax2.set_title("Shape from Texture")

    ax3.plot_surface(X, Y, defocus_depth, cmap='gray')
    ax3.set_title("Shape from Defocus")

    plt.show()
