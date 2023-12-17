import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from siftinator import compute_descriptors
import time


def match_image(match_args):
    ref_descriptor, descriptors, bf = match_args
    matches = bf.match(ref_descriptor, descriptors)
    return len(matches)


def plot_matches(matches, reference_images_path):
    jpg_files = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')]
    fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    axes[0, 0].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[matches[5]]}"))
    axes[0, 1].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[matches[4]]}"))
    axes[0, 2].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[matches[3]]}"))
    axes[1, 0].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[matches[2]]}"))
    axes[1, 1].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[matches[1]]}"))
    axes[1, 2].imshow(plt.imread(f"flowers/flowers-102/jpg/{jpg_files[matches[0]]}"))
    for ax in axes.flatten():
        ax.axis('off')
    plt.show()


def find_closest_images(ref_descriptor, loaded_descriptors, bf):
    with ThreadPoolExecutor() as executor:
        args_list = [(ref_descriptor, descriptors, bf) for descriptors in loaded_descriptors]
        images_matches = list(executor.map(match_image, args_list))

    closest_images = np.argsort(images_matches)[-6:]
    return closest_images


def main(descriptors_path, query_image_path, reference_images_path, clip):
    start_time = time.time()
    ref_descriptor = compute_descriptors(query_image_path)
    loaded_descriptors = np.load(descriptors_path, allow_pickle=True)[:clip]
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    closest_images = find_closest_images(ref_descriptor, loaded_descriptors, bf)
    print(f"{time.time() - start_time:.2f}")
    plot_matches(closest_images, reference_images_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Matching Script")
    parser.add_argument("--descriptors_path", type=str, help="Path to the descriptors file")
    parser.add_argument("--query_image_path", type=str, help="Path to the query image")
    parser.add_argument("--reference_images_path", type=str, help="Path to the reference images")
    parser.add_argument("--clip", type=int, help="Number of images to process")

    args = parser.parse_args()
    main(args.descriptors_path, args.query_image_path, args.reference_images_path, args.clip)
