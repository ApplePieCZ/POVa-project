# POVa Project - CBIR
# Lukas Marek
# 20.12.2023
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
                                         'sift.py',
                                         '-i', 'flowers/flowers-102/jpg/image_00001.jpg',
                                         '-d', 'flowers/flowers-102/jpg',
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


def plot_resnet():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    resnet_y = [0.97, 1.59, 3.12, 5.15, 8.12, 14.20, 26.16]
    effnet_y = [0.99, 1.57, 2.66, 4.76, 8.02, 14.32, 26.79]
    alexnet_y = [0.81, 1.29, 2.09, 3.66, 6.81, 12.92, 25.36]
    resnet_pre_y = [0.32, 0.33, 0.33, 0.36, 0.37, 0.45, 0.54]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, resnet_y, marker='o', linestyle='--', color='red', label='ResNet50')
    ax.plot(x, effnet_y, marker='o', linestyle='--', color='green', label='EfficientNetV2')
    ax.plot(x, alexnet_y, marker='o', linestyle='--', color='gold', label='AlexNet')
    ax.plot(x, resnet_pre_y, marker='o', linestyle='--', color='crimson', label='ResNet50 uložené features')
    ax.set_xlabel('Obrázky')
    ax.set_ylabel('Čas (sekundy)')
    ax.set_title('Neuronové sítě')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_resnet_orb():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    resnet_y = [1.33, 1.92, 3.28, 5.43, 8.44, 14.55, 26.62]
    resnet_pre_y = [0.32, 0.33, 0.33, 0.36, 0.37, 0.45, 0.54]
    orb_y = [0.43, 0.70, 1.19, 2.13, 4.13, 8.08, 16.03]
    orb_knn = [0.42, 0.60, 1.02, 1.70, 3.24, 6.31, 12.57]
    orb_flann = [0.33, 0.50, 0.79, 1.38, 2.57, 5.00, 9.87]
    orb_pre_y = [0.38, 0.52, 0.78, 1.29, 2.34, 4.47, 8.72]
    orb_flann_pre = [0.31, 0.33, 0.42, 0.52, 0.72, 1.20, 2.13]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, resnet_y, marker='o', linestyle='--', color='red', label='ResNet50')
    ax.plot(x, orb_y, marker='o', linestyle='--', color='red', label='ORB BFMatch')
    ax.plot(x, orb_knn, marker='o', linestyle='--', color='green', label='ORB BF knnMatch')
    ax.plot(x, orb_flann, marker='o', linestyle='--', color='blue', label='ORB Flann knnMatch')
    ax.plot(x, orb_pre_y, marker='o', linestyle='--', color='orange', label='ORB Uložené rysy')
    ax.plot(x, orb_flann_pre, marker='o', linestyle='--', color='crimson', label='ORB Flann uložené rysy')
    ax.plot(x, resnet_pre_y, marker='o', linestyle='--', color='orange', label='ResNet50 uložené features')
    ax.set_xlabel('Obrázky')
    ax.set_ylabel('Čas (sekundy)')
    ax.set_title('Porovnání Resnet50 a ORB')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_orb():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    orb_y = [0.43, 0.70, 1.19, 2.13, 4.13, 8.08, 16.03]
    orb_knn = [0.42, 0.60, 1.02, 1.70, 3.24, 6.31, 12.57]
    orb_flann = [0.33, 0.50, 0.79, 1.38, 2.57, 5.00, 9.87]
    orb_pre_y = [0.38, 0.52, 0.78, 1.29, 2.34, 4.47, 8.72]
    orb_flann_pre = [0.31, 0.33, 0.42, 0.52, 0.72, 1.20, 2.13]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, orb_y, marker='o', linestyle='--', color='red', label='BFMatch')
    ax.plot(x, orb_knn, marker='o', linestyle='--', color='green', label='BF knnMatch')
    ax.plot(x, orb_flann, marker='o', linestyle='--', color='blue', label='Flann knnMatch')
    ax.plot(x, orb_pre_y, marker='o', linestyle='--', color='orange', label='Uložené rysy')
    ax.plot(x, orb_flann_pre, marker='o', linestyle='--', color='crimson', label='Flann uložené rysy')
    ax.set_xlabel('Obrázky')
    ax.set_ylabel('Čas (sekundy)')
    ax.set_title('ORB')
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_sift():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    sift_y = [9.83, 18.90, 26.02, 39.45, 75.13, 162.09, 354.66]
    sift_pre_y = [9.84, 17.60, 22.90, 32.28, 59.13, 128.37, 285.30]
    sift_knn = [3.31, 6.21, 9.96, 17.10, 33.77, 69.42, 143.51]
    sift_flann = [3.01, 5.66, 9.52, 16.80, 33.48, 67.93, 139.47]
    sift_flann_pre = [2.45, 3.19, 3.81, 5.15, 8.40, 15.41, 30.68]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, sift_y, marker='o', linestyle='--', color='red', label='BFMatch')
    ax.plot(x, sift_pre_y, marker='o', linestyle='--', color='darkred', label='BFMatch uložené rysy')
    ax.plot(x, sift_knn, marker='o', linestyle='--', color='orange', label='BF knnMatch')
    ax.plot(x, sift_flann, marker='o', linestyle='--', color='green', label='Flann knnMatch')
    ax.plot(x, sift_flann_pre, marker='o', linestyle='--', color='darkgreen', label='Flann uložené rysy')
    ax.set_xlabel('Obrázky')
    ax.set_ylabel('Čas (sekundy)')
    ax.set_title('SIFT')
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


def plot_sift_orb():
    x = [128, 256, 512, 1024, 2048, 4096, 8192]
    sift_y = [9.83, 18.90, 26.02, 39.45, 75.13, 162.09, 354.66]
    sift_pre_y = [9.84, 17.60, 22.90, 32.28, 59.13, 128.37, 285.30]
    sift_knn = [3.31, 6.21, 9.96, 17.10, 33.77, 69.42, 143.51]
    sift_flann = [3.01, 5.66, 9.52, 16.80, 33.48, 67.93, 139.47]
    sift_flann_pre = [2.45, 3.19, 3.81, 5.15, 8.40, 15.41, 30.68]
    orb_y = [0.43, 0.70, 1.19, 2.13, 4.13, 8.08, 16.03]
    orb_knn = [0.42, 0.60, 1.02, 1.70, 3.24, 6.31, 12.57]
    orb_flann = [0.33, 0.50, 0.79, 1.38, 2.57, 5.00, 9.87]
    orb_pre_y = [0.38, 0.52, 0.78, 1.29, 2.34, 4.47, 8.72]
    orb_flann_pre = [0.31, 0.33, 0.42, 0.52, 0.72, 1.20, 2.13]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x, sift_y, marker='o', linestyle='--', color='red', label='SIFT BFMatch')
    ax.plot(x, sift_knn, marker='o', linestyle='--', color='orange', label='SIFT BF knnMatch')
    ax.plot(x, sift_flann, marker='o', linestyle='--', color='green', label='SIFT Flann knnMatch')
    ax.plot(x, sift_pre_y, marker='o', linestyle='--', color='darkred', label='SIFT BFMatch uložené rysy')
    ax.plot(x, sift_flann_pre, marker='o', linestyle='--', color='darkgreen', label='SIFT Flann uložené rysy')
    ax.plot(x, orb_y, marker='o', linestyle='--', color='violet', label='ORB BFMatch')
    ax.plot(x, orb_pre_y, marker='o', linestyle='--', color='purple', label='ORB Uložené rysy')
    ax.plot(x, orb_knn, marker='o', linestyle='--', color='pink', label='ORB BF knnMatch')
    ax.plot(x, orb_flann, marker='o', linestyle='--', color='crimson', label='ORB Flann knnMatch')
    ax.plot(x, orb_flann_pre, marker='o', linestyle='--', color='gold', label='ORB Flann uložené rysy')
    ax.set_xlabel('Obrázky')
    ax.set_ylabel('Čas (sekundy)')
    ax.set_title('Porovnání SIFT a ORB')
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_resnet()
