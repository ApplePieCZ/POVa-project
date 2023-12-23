import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from siftinator import compute_descriptors


def process_image(image, reference_images_path):
    descriptors = compute_descriptors(os.path.join(reference_images_path, image))
    return descriptors


def generate_descriptors(reference_images_path, output_path):
    jpg_files = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')]
    descriptors_list = []

    with ThreadPoolExecutor() as executor:
        descriptors_list = list(executor.map(process_image, jpg_files, [reference_images_path]*len(jpg_files)))

    np.save(output_path, np.array(descriptors_list, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Descriptors Script")
    parser.add_argument("--reference_images_path", type=str, help="Path to reference images")
    parser.add_argument("--output_path", type=str, help="Path to save output descriptors")

    args = parser.parse_args()

    generate_descriptors(args.reference_images_path, args.output_path)