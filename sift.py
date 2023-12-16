import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor


def compute_sift(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    image_keypoints, image_descriptors = sift.detectAndCompute(gray, None)
    return image_keypoints, image_descriptors


def process_image(image):
    _, descriptors = compute_sift(f"{reference_images_path}/{image}")
    matches = bf.match(query_descriptors, descriptors)
    return len(matches)


if __name__ == "__main__":
    query_image_path = 'jpg/image_00001.jpg'
    reference_images_path = 'flowers/flowers-102/jpg'
    start_time = time.time()

    query_image = cv2.imread(query_image_path)
    _, query_descriptors = compute_sift(query_image_path)

    jpg_files = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')][:1000]

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    #images_matches = []

    with ThreadPoolExecutor() as executor:
        # Process images in parallel and collect descriptors
        images_matches = list(executor.map(process_image, jpg_files))

    #np.save("descriptors.npy", np.array(descriptors_list, dtype=object))
    print(time.time() - start_time)

    closest_images = np.argsort(images_matches)[-6:]

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



