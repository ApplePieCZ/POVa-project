# CBIR using Pytorch library
**ResNet50**  

python resnet.py -i [path/to/image] -d  [path/to/image/folder] -r [dataset/root] -c [number of images] -f
- -i -- input image for search
- -d -- folder with all images that will be searched
- -r -- root folder of dataset
- -c -- number of images to be processed (0+)
- -f -- to use saved vectors, use this argument

**SIFT and ORB**  

python sift.py -i [path/to/image] -d  [path/to/image/folder] -c [number of images] -s
- -i --  input image for search
- -d -- folder with all images that will be searched
- -c -- number of images to be processed (0+)
- -s -- to use SIFT instead od ORB, use this argument

**Autoencoder**   

$python ./main.py
- for training, set global variable TRAIN to True in main.py 
- for Comparison, set TRAIN to False and ONLY_PROCESS to False 
- for Comparison with already encoded database set TRAIN to False and ONLY_PROCESS to True.

