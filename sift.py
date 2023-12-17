# POVa Project - CBIR
# Lukas Marek, with help of Tomas Krsicka
# 17.12.2023
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import argparse
import time


def compute(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_descriptors = algo.detectAndCompute(gray, None)
    return image_descriptors


def process_image(image):
    descriptors = compute(f"{reference_images_path}/{image}")
    matches = bf.match(query_descriptors, descriptors)
    return len(matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIFT/ORB based CBIR.")
    parser.add_argument("-i", type=str, help="Query image")
    parser.add_argument("-d", type=str, help="Path to reference images")
    parser.add_argument("-c", type=int, help="Size of dataset to clip")
    parser.add_argument("-s", action='store_true', help="If SIFT is used.")

    args = parser.parse_args()

    if args.s:
        algo = cv2.SIFT_create()
    else:
        algo = cv2.ORB_create()

    if args.c:
        clip = args.c
    else:
        clip = 1000

    query_image_path = args.i
    reference_images_path = args.d

    start_time = time.time()
    query_image = cv2.imread(query_image_path)
    query_descriptors = compute(query_image_path)

    jpg_files = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')][:clip]

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    with ThreadPoolExecutor() as executor:
        images_matches = list(executor.map(process_image, jpg_files))

    print(f"{time.time() - start_time:.2f}")

    closest_images = np.argsort(images_matches)[-6:]

    fig, axes = plt.subplots(2, 3, figsize=(8, 8))

    axes[0, 0].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[5]]}"))
    axes[0, 1].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[4]]}"))
    axes[0, 2].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[3]]}"))
    axes[1, 0].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[2]]}"))
    axes[1, 1].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[1]]}"))
    axes[1, 2].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[0]]}"))

    for ax in axes.flatten():
        ax.axis('off')

    plt.show()



