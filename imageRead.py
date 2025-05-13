import cv2

# Load image
image = cv2.imread("testImage.jpg")

# Display image
if image is None:
    print("Image not found or can't be opened.")
cv2.imshow("My test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()