import cv2


def compute_descriptors(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, image_descriptors = sift.detectAndCompute(gray, None)
    return image_descriptors