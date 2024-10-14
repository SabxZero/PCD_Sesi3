import imageio
import numpy as np
import matplotlib.pyplot as plt

image_path = 'RGB.jpg'  
image_rgb = imageio.imread(image_path)

image_gray = np.dot(image_rgb[..., :3], [0.2989, 0.5870, 0.1140])

histogram, bin_edges = np.histogram(image_gray, bins=256, range=(0, 255))

total_pixels = np.sum(histogram)

dominant_intensity = np.argmax(histogram)
dominant_count = histogram[dominant_intensity]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.axis('off')
plt.title("Gambar Grayscale")

plt.subplot(1, 2, 2)
plt.title("Histogram Gambar Grayscale")
plt.xlabel("Intensitas Piksel (0-255)")
plt.ylabel("Frekuensi")
plt.xlim([0, 255])
plt.bar(bin_edges[0:-1], histogram, width=1, color='black')

plt.tight_layout()
plt.show()

print(f"Total jumlah piksel: {total_pixels}")
print(f"Intensitas dominan: {dominant_intensity} dengan jumlah piksel: {dominant_count}")
