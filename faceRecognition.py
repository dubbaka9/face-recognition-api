import face_recognition
import cv2

print("Face recognition is working!")

image = face_recognition.load_image_file("/Users/sindhu/Git/face-recognition-api/testImage.jpg")
face_locations = face_recognition.face_locations(image)
print("Found", len(face_locations), "faces.")

# # Optional: Draw rectangles with OpenCV
for (top, right, bottom, left) in face_locations:
#    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    center_x = left + (right - left) // 2
    center_y = top + (bottom - top) // 2

    # Calculate axes (radii) for width and height
    axes_length = ((right - left) // 2, (bottom - top) // 2)

    # Draw ellipse
    cv2.ellipse(image, (center_x, center_y), axes_length, 0, 0, 360, (0, 255, 0), 2)


cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
