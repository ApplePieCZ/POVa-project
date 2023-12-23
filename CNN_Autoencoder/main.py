from Autoencoder import TN_Autoencoder
from Inference import inference
from Train import train

TRAIN = True
ONLY_PROCESS = False
IMPATH= r"flowers\flowers-102\jpg\image_04947.jpg"


SOURCE_IMGS=[
r"flowers\flowers-102\jpg\image_00001.jpg",
r"flowers\flowers-102\jpg\image_00428.jpg",
r"flowers\flowers-102\jpg\image_00542.jpg",
r"flowers\flowers-102\jpg\image_01104.jpg",
r"flowers\flowers-102\jpg\image_02660.jpg",
r"flowers\flowers-102\jpg\image_03994.jpg"]


def main():
    autoencoder = TN_Autoencoder
    if TRAIN:
        train(autoencoder)
    else:
        inference(autoencoder, ONLY_PROCESS, SOURCE_IMGS)



if __name__ == "__main__":
    main()
