# POVa Project - CBIR
# Lukas Marek, with help of Tomas Krsicka
# 20.12.2023
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


def process_image_norm(image):
    descriptors = compute(f"{reference_images_path}/{image}")
    matches = bf_norm.match(query_descriptors, descriptors)
    return len(matches)


def process_image_knn(image):
    descriptors = compute(f"{reference_images_path}/{image}")
    matches = bf.knnMatch(query_descriptors, descriptors, k=2)
    a = 0
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            a += 1
    return a


def process_image_flann_orb(image):
    descriptors = compute(f"{reference_images_path}/{image}")
    matches = flann_orb.knnMatch(query_descriptors, descriptors, k=2)
    a = 0
    for i, match in enumerate(matches):
        if len(match) >= 2:
            m, n = match
            if m.distance < 0.95 * n.distance:
                a += 1
    return a


def process_image_flann(image):
    descriptors = compute(f"{reference_images_path}/{image}")
    matches = flann.knnMatch(query_descriptors, descriptors, k=2)
    a = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.95 * n.distance:
            a += 1
    return a


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

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    FLANN_INDEX_LSH = 6
    index2_params = dict(algorithm=FLANN_INDEX_LSH,
                         table_number=12,  # 12
                         key_size=20,  # 20
                         multi_probe_level=2)  # 2
    flann_orb = cv2.FlannBasedMatcher(index2_params, search_params)

    start_time = time.time()
    query_image = cv2.imread(query_image_path)
    query_descriptors = compute(query_image_path)

    jpg_files = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')][:clip]

    bf_norm = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    bf = cv2.BFMatcher()

    with ThreadPoolExecutor() as executor:
        images_matches = list(executor.map(process_image_flann_orb, jpg_files))

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
