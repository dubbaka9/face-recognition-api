import face_recognition
import cv2
import numpy as np
from fastapi.responses import JSONResponse

print("Face recognition module loaded!")

def recognise_faces(image_path):
    """
    Recognize and locate faces in an image
    Display the image with face rectangles and return JSON response
    """
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image)
        
        print(f"Found {len(face_locations)} faces.")
        
        # Convert RGB to BGR for OpenCV display
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw rectangles around detected faces and add labels
        faces_data = []
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw rectangle around face
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add face ID label
            label = f"Face {i + 1}"
            cv2.putText(image_bgr, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Store face information
            face_info = {
                "face_id": i + 1,
                "location": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left)
                },
                "width": int(right - left),
                "height": int(bottom - top)
            }
            faces_data.append(face_info)
        
        # Display the image with detected faces
        window_name = "Face Recognition Results"
        cv2.imshow(window_name, image_bgr)
        
        # Add instructions for user
        print("Press any key to close the image window...")
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyWindow(window_name)
        
        return {
            "success": True,
            "faces_count": len(face_locations),
            "faces": faces_data,
            "message": f"Successfully detected {len(face_locations)} face(s). Image displayed with face markers."
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Face recognition failed: {str(e)}"}
        )

def recognise_faces_with_display(image_path):
    """
    Recognize faces and display them with rectangles (for local testing)
    """
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw rectangles around faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            
        # Display image
        cv2.imshow("Detected Faces", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return len(face_locations)
        
    except Exception as e:
        print(f"Error in face recognition with display: {str(e)}")
        return 0