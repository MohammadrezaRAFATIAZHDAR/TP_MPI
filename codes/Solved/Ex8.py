from mpi4py import MPI
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

M = 255

# First method for stretching contrast
def f_one(x, n):
    if x == 0:
        return 0
    return int(M**(1-n) * (x**n))

# Second method for stretching contrast
def f_two(x, n):
    if x == 0:
        return 0
    return int((M**((n-1)/n)) * (x**(1/n)))

# Converts an image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Loads an image and converts it to grayscale (rank 0 only)
def readImage():
    img = mpimg.imread('image.png')
    plt.imshow(img)
    print("Press 'q' to continue")
    plt.show()
    grey = rgb2gray(img)
    pixels = (np.ravel(grey) * 255).astype(np.int32)
    return pixels, len(grey), len(grey[0])

# Saves the processed image (rank 0 only)
def saveImage(newP, nblines, nbcolumns):
    newimg = newP.reshape((nblines, nbcolumns))
    plt.imshow(newimg, cmap=cm.Greys_r)
    print("Press 'q' to continue")
    plt.show()
    mpimg.imsave('image-grey2-stretched.png', newimg, cmap=cm.Greys_r)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Load image on rank 0
if rank == 0:
    pixels, nblines, nbcolumns = readImage()
    total_pixels = len(pixels)
else:
    pixels = None
    nblines = nbcolumns = total_pixels = 0

start_time = MPI.Wtime()
# Broadcast image dimensions to all processes
nblines = comm.bcast(nblines, root=0)
nbcolumns = comm.bcast(nbcolumns, root=0)
total_pixels = comm.bcast(total_pixels, root=0)

# Scatter pixel data to all processes
local_pixels = np.empty(total_pixels // size, dtype='i')
comm.Scatter(pixels, local_pixels, root=0)

# Compute local min and max
local_min = np.min(local_pixels)
local_max = np.max(local_pixels)

# Reduce to find global min and max
pix_min = comm.reduce(local_min, op=MPI.MIN, root=0)
pix_max = comm.reduce(local_max, op=MPI.MAX, root=0)

# Broadcast global min and max to all processes
pix_min = comm.bcast(pix_min, root=0)
pix_max = comm.bcast(pix_max, root=0)

# Compute alpha
alpha = 1 + (pix_max - pix_min) / M

# Stretch contrast locally
for i in range(len(local_pixels)):
    if rank % 2 == 0:
        local_pixels[i] = f_one(local_pixels[i], alpha)
    else:
        local_pixels[i] = f_two(local_pixels[i], alpha)

# Gather processed pixel data on rank 0
processed_pixels = None
if rank == 0:
    processed_pixels = np.empty(total_pixels, dtype='i')

comm.Gather(local_pixels, processed_pixels, root=0)
stop_time = MPI.Wtime()
# Save the processed image on rank 0
if rank == 0:
    saveImage(processed_pixels, nblines, nbcolumns)
    print("Stretching done...")
    print(f"Stretching done in {stop_time - start_time:.4f} seconds with {size} processes.")
