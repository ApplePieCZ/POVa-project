# --------------------------------------------------------------------------------------------------------------
# Code skeleton based on https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/
# --------------------------------------------------------------------------------------------------------------
from Autoencoder import NFS_autoencoder
from Inference import inference
from Train import train

TRAIN = False
ONLY_PROCESS = True
IMPATH= r"flowers\flowers-102\jpg\image_04947.jpg"


SOURCE_IMGS=[r""]

def main():
    autoencoder = NFS_autoencoder
    if TRAIN:
        train(autoencoder)
    else:
        inference(autoencoder, ONLY_PROCESS, [IMPATH])



if __name__ == "__main__":
    main()
