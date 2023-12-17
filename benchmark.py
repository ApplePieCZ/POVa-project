# POVa Project - CBIR
# Lukas Marek
# 17.12.2023
import subprocess
import re
import matplotlib.pyplot as plt


def resnet():
    print("*--------------RESNET----------------*")

    for value in [128, 256, 512, 1024, 2048, 4096, 8192]:
        time_medium = 0.0
        for i in range(3):
            try:
                result = subprocess.run(['C:\\Users\\lukas\\PycharmProjects\\pythonProject\\venv\\Scripts\\python',
                                         'resnet.py',
                                         '-i', 'flowers/flowers-102/jpg/image_00001.jpg',
                                         '-d', 'flowers/flowers-102/jpg',
                                         '-r', 'flowers',
                                         '-c', str(value)
                                         ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running the script: {e}")
                print("Error output:", e.stderr)
                exit()
            cleaned_output = re.sub(r'\x1b\[[0-9;]*[mGK]', '', result.stdout)
            cleaned_output = cleaned_output.strip()
            time_medium += float(cleaned_output)
        time_medium = time_medium / 3
        print(f"Number of images: {value} || Time: {time_medium:.2f}")
    print("*-------------------------------------*")


def sift():
    print("*--------------SIFT----------------*")

    for value in [128, 256, 512, 1024, 2048, 4096, 8192]:
        time_medium = 0.0
        for i in range(3):
            try:
                result = subprocess.run(['C:\\Users\\lukas\\PycharmProjects\\pythonProject\\venv\\Scripts\\python',
                                         'sift-VF.py',
                                         '-i', 'flowers/flowers-102/jpg/image_00001.jpg',
                                         '-d', 'flowers/flowers-102/jpg',
                                         '-c', str(value),
                                         '-s'
                                         ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running the script: {e}")
                print("Error output:", e.stderr)
                exit()
            cleaned_output = re.sub(r'\x1b\[[0-9;]*[mGK]', '', result.stdout)
            cleaned_output = cleaned_output.strip()
            time_medium += float(cleaned_output)
        time_medium = time_medium / 3
        print(f"Number of images: {value} || Time: {time_medium:.2f}")
    print("*-------------------------------------*")


def plot_resnet():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    resnet_y = [1.33, 1.92, 3.28, 5.43, 8.44, 14.55, 26.62]
    resnet_pre_y = [0.32, 0.33, 0.33, 0.36, 0.37, 0.45, 0.54]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, resnet_y, marker='o', linestyle='--', color='red', label='Unprepared features')
    ax.plot(x, resnet_pre_y, marker='o', linestyle='--', color='orange', label='Prepared features')
    ax.set_xlabel('Images')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Resnet')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_orb():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    orb_y = [0.43, 0.70, 1.19, 2.13, 4.13, 8.08, 16.03]
    orb_pre_y = [0.38, 0.52, 0.78, 1.29, 2.34, 4.47, 8.72]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, orb_y, marker='o', linestyle='--', color='red', label='Unprepared features')
    ax.plot(x, orb_pre_y, marker='o', linestyle='--', color='orange', label='Prepared features')
    ax.set_xlabel('Images')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Resnet')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_sift():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    sift_y = [9.83, 18.90, 26.02, 39.45, 75.13, 162.09, 354.66]
    sift_pre_y = [9.84, 17.60, 22.90, 32.28, 59.13, 128.37, 285.30]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, sift_y, marker='o', linestyle='--', color='darkred', label='Unprepared features')
    ax.plot(x, sift_pre_y, marker='o', linestyle='--', color='orange', label='Prepared features')
    ax.set_xlabel('Images')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Resnet')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_all():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    resnet_y = [1.33, 1.92, 3.28, 5.43, 8.44, 14.55, 26.62]
    resnet_pre_y = [0.32, 0.33, 0.33, 0.36, 0.37, 0.45, 0.54]
    sift_y = [9.83, 18.90, 26.02, 39.45, 75.13, 162.09, 354.66]
    sift_pre_y = [9.84, 17.60, 22.90, 32.28, 59.13, 128.37, 285.30]
    orb_y = [0.43, 0.70, 1.19, 2.13, 4.13, 8.08, 16.03]
    orb_pre_y = [0.38, 0.52, 0.78, 1.29, 2.34, 4.47, 8.72]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, resnet_y, marker='o', linestyle='--', color='darkred', label='Unprepared Resnet')
    ax.plot(x, resnet_pre_y, marker='o', linestyle='--', color='red', label='Prepared Resnet')
    ax.plot(x, sift_y, marker='o', linestyle='--', color='darkgreen', label='Unprepared SIFT')
    ax.plot(x, sift_pre_y, marker='o', linestyle='--', color='green', label='Prepared SIFT')
    ax.plot(x, orb_y, marker='o', linestyle='--', color='darkorange', label='Unprepared ORB')
    ax.plot(x, orb_pre_y, marker='o', linestyle='--', color='orange', label='Prepared ORB')
    ax.set_xlabel('Images')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time comparison between ResNet, SIFT and ORB')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_two():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    resnet_y = [1.33, 1.92, 3.28, 5.43, 8.44, 14.55, 26.62]
    resnet_pre_y = [0.32, 0.33, 0.33, 0.36, 0.37, 0.45, 0.54]
    orb_y = [0.43, 0.70, 1.19, 2.13, 4.13, 8.08, 16.03]
    orb_pre_y = [0.38, 0.52, 0.78, 1.29, 2.34, 4.47, 8.72]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, resnet_y, marker='o', linestyle='--', color='darkred', label='Unprepared Resnet')
    ax.plot(x, resnet_pre_y, marker='o', linestyle='--', color='red', label='Prepared Resnet')
    ax.plot(x, orb_y, marker='o', linestyle='--', color='darkorange', label='Unprepared ORB')
    ax.plot(x, orb_pre_y, marker='o', linestyle='--', color='orange', label='Prepared ORB')
    ax.set_xlabel('Images')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time comparison between ResNet and ORB')
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_all()
    plot_two()
