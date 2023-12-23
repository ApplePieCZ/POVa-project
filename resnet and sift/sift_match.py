import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from siftinator import compute_descriptors
import time

#1, 7095, 1971, 1845, 1415, 4516, 7455

def match_image(match_args):
    ref_descriptor, descriptors, bf = match_args
    #matches = bf.match(ref_descriptor, descriptors)
    #return len(matches)
    matches = flann_orb.knnMatch(ref_descriptor, descriptors, k=2)
    a = 0
    for i, match in enumerate(matches):
        if len(match) >= 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                a += 1
    return a


def match_image_sift(match_args):
    ref_descriptor, descriptors, bf = match_args
    matches = bf.match(ref_descriptor, descriptors)
    return len(matches)
    #matches = flann.knnMatch(ref_descriptor, descriptors, k=2)
    #a = 0
    #for i, (m, n) in enumerate(matches):
    #    if m.distance < 0.94 * n.distance:
    #        a += 1
    #return a
    #matches = bf.knnMatch(ref_descriptor, descriptors, k=2)
    #a = 0
    #for m, n in matches:
    #    if m.distance < 0.95 * n.distance:
    #        a += 1
    #return a


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
        images_matches = list(executor.map(match_image_sift, args_list))

    closest_images = np.argsort(images_matches)[-6:]
    return closest_images


def main(descriptors_path, query_image_path, reference_images_path, clip):
    loaded_descriptors = np.load(descriptors_path, allow_pickle=True)[:clip]
    start_time = time.time()
    ref_descriptor = compute_descriptors(query_image_path)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    #bf = cv2.BFMatcher()
    closest_images = find_closest_images(ref_descriptor, loaded_descriptors, bf)
    print(f"{time.time() - start_time:.2f}")
    plot_matches(closest_images, reference_images_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Matching Script")
    parser.add_argument("--descriptors_path", type=str, help="Path to the descriptors file")
    parser.add_argument("--query_image_path", type=str, help="Path to the query image")
    parser.add_argument("--reference_images_path", type=str, help="Path to the reference images")
    parser.add_argument("--clip", type=int, help="Number of images to process")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    FLANN_INDEX_LSH = 6
    index2_params = dict(algorithm=FLANN_INDEX_LSH,
                         table_number=6,  # 12
                         key_size=12,  # 20
                         multi_probe_level=1)  # 2
    search_params = dict(checks=50)
    flann_orb = cv2.FlannBasedMatcher(index2_params, search_params)

    args = parser.parse_args()
    main(args.descriptors_path, args.query_image_path, args.reference_images_path, args.clip)
