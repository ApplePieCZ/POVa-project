# POVa Project - CBIR
# Lukas Marek
# 17.12.2023
import subprocess
import os
import time
import re

if __name__ == "__main__":
    command = [
        'C:\\Users\\lukas\\PycharmProjects\\pythonProject\\venv\\Scripts\\python',  # Python interpreter
        'resnet.py',  # Your script name
        '-i', 'flowers/flowers-102/jpg/image_00001.jpg',  # Input file
        '-d', 'flowers/flowers-102/jpg',  # Input directory
        '-r', 'flowers',  # Output directory
        '-c', '100'  # Some other argument
    ]

    print("*--------------RESNET----------------*")

    for value in [128, 256, 512]: #, 1024, 2048, 4096, 8192]:
        time_medium = 0.0
        for i in range(3):
            try:
                result = subprocess.run(['C:\\Users\\lukas\\PycharmProjects\\pythonProject\\venv\\Scripts\\python',
                                         'resnet.py',  # Your script name
                                         '-i', 'flowers/flowers-102/jpg/image_00001.jpg',  # Input file
                                         '-d', 'flowers/flowers-102/jpg',  # Input directory
                                         '-r', 'flowers',  # Output directory
                                         '-c', str(value)  # Some other argument
                                         ], check=True, capture_output=True, text=True)
                # Access the output, result.stdout
            except subprocess.CalledProcessError as e:
                print(f"Error running the script: {e}")
                # Access the error, e.stderr
                print("Error output:", e.stderr)
                exit()
            cleaned_output = re.sub(r'\x1b\[[0-9;]*[mGK]', '', result.stdout)
            cleaned_output = cleaned_output.strip()
            time_medium += float(cleaned_output)
        time_medium = time_medium / 3
        print(f"Number of images: {value} || Time: {time_medium:.2f}")

    print("*-------------------------------------*")
