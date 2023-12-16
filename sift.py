import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#SIFT COMPUTE
def compute_sift(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    image_keypoints, image_descriptors = sift.detectAndCompute(gray, None)
    return image_keypoints, image_descriptors


if __name__ == "__main__":
    query_image_path = 'jpg/image_00001.jpg'
    reference_images_path = 'flowers/flowers-102/jpg'

    query_image = cv2.imread(query_image_path)
    query_keypoints, query_descriptors = compute_sift(query_image_path)

    jpg_files = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')][:500]

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    images_matches = []

    for image in jpg_files:
        keypoints, descriptors = compute_sift(f"{reference_images_path}/{image}")
        matches = bf.match(query_descriptors, descriptors)
        images_matches.append(len(matches))

    closest_images = np.argsort(images_matches)[-6:]
    print(closest_images)

    fig, axes = plt.subplots(2, 3, figsize=(8, 8))

    axes[0, 0].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[closest_images[5]]}"))
    axes[0, 1].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[closest_images[4]]}"))
    axes[0, 2].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[closest_images[3]]}"))
    axes[1, 0].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[closest_images[2]]}"))
    axes[1, 1].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[closest_images[1]]}"))
    axes[1, 2].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[closest_images[0]]}"))

    for ax in axes.flatten():
        ax.axis('off')

    plt.show()



