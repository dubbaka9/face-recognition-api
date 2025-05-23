import face_recognition
import cv2
import numpy as np
from fastapi.responses import JSONResponse

def compare_faces(known_image_path: str, unknown_image_path: str):
    """
    Compare faces in two images and display them with highlighted matching faces
    """
    try:
        # Load and encode known image
        known_image = face_recognition.load_image_file(known_image_path)
        known_encodings = face_recognition.face_encodings(known_image)
        if not known_encodings:
            return JSONResponse(status_code=400, content={"error": "No face found in known image."})
        
        # Load and encode unknown image
        unknown_image = face_recognition.load_image_file(unknown_image_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        if not unknown_encodings:
            return JSONResponse(status_code=400, content={"error": "No face found in unknown image."})

        # Get face locations for display
        known_face_locations = face_recognition.face_locations(known_image)
        unknown_face_locations = face_recognition.face_locations(unknown_image)

        # Convert images to BGR for OpenCV
        known_image_bgr = cv2.cvtColor(known_image, cv2.COLOR_RGB2BGR)
        unknown_image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

        # Find matches between all faces
        matches_found = []
        best_match = None
        best_distance = float('inf')

        for i, known_encoding in enumerate(known_encodings):
            for j, unknown_encoding in enumerate(unknown_encodings):
                # Compare faces
                result = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6)[0]
                distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
                
                if result:
                    matches_found.append({
                        'known_face_index': i,
                        'unknown_face_index': j,
                        'distance': float(distance),
                        'confidence': float(1 - distance)
                    })
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = {
                            'known_face_index': i,
                            'unknown_face_index': j,
                            'distance': float(distance),
                            'confidence': float(1 - distance)
                        }

        # Prepare response data
        response_data = {
            "match": len(matches_found) > 0,
            "matches_count": len(matches_found),
            "best_match": best_match,
            "all_matches": matches_found
        }

        if best_match:
            response_data.update({
                "confidence_score": best_match['confidence'],
                "distance": best_match['distance']
            })

        # Display images with highlighted faces if matches found
        if matches_found:
            display_matched_faces(
                known_image_bgr, unknown_image_bgr,
                known_face_locations, unknown_face_locations,
                matches_found
            )
            response_data["message"] = f"Found {len(matches_found)} matching face(s). Images displayed with highlighted matches."
        else:
            # Still display images but with different colors for no matches
            display_no_matches(
                known_image_bgr, unknown_image_bgr,
                known_face_locations, unknown_face_locations
            )
            response_data["message"] = "No matching faces found. Images displayed for comparison."

        return response_data

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def display_matched_faces(known_image, unknown_image, known_locations, unknown_locations, matches):
    """
    Display both images side by side with matching faces highlighted
    """
    # Get image dimensions
    known_height, known_width = known_image.shape[:2]
    unknown_height, unknown_width = unknown_image.shape[:2]
    
    # Create a combined image (side by side)
    max_height = max(known_height, unknown_height)
    combined_width = known_width + unknown_width + 20  # 20px gap
    
    # Create white background
    combined_image = np.ones((max_height, combined_width, 3), dtype=np.uint8) * 255
    
    # Place images
    combined_image[0:known_height, 0:known_width] = known_image
    combined_image[0:unknown_height, known_width + 20:known_width + 20 + unknown_width] = unknown_image
    
    # Draw rectangles around all faces first (gray for unmatched)
    for i, (top, right, bottom, left) in enumerate(known_locations):
        color = (128, 128, 128)  # Gray for unmatched
        cv2.rectangle(combined_image, (left, top), (right, bottom), color, 2)
        cv2.putText(combined_image, f"K{i+1}", (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    for i, (top, right, bottom, left) in enumerate(unknown_locations):
        color = (128, 128, 128)  # Gray for unmatched
        x_offset = known_width + 20
        cv2.rectangle(combined_image, (left + x_offset, top), (right + x_offset, bottom), color, 2)
        cv2.putText(combined_image, f"U{i+1}", (left + x_offset, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Highlight matching faces in green
    for match in matches:
        known_idx = match['known_face_index']
        unknown_idx = match['unknown_face_index']
        confidence = match['confidence']
        
        # Highlight known face
        if known_idx < len(known_locations):
            top, right, bottom, left = known_locations[known_idx]
            cv2.rectangle(combined_image, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(combined_image, f"MATCH K{known_idx+1}", (left, top - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined_image, f"{confidence:.2f}", (left, bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Highlight unknown face
        if unknown_idx < len(unknown_locations):
            top, right, bottom, left = unknown_locations[unknown_idx]
            x_offset = known_width + 20
            cv2.rectangle(combined_image, (left + x_offset, top), (right + x_offset, bottom), (0, 255, 0), 3)
            cv2.putText(combined_image, f"MATCH U{unknown_idx+1}", (left + x_offset, top - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined_image, f"{confidence:.2f}", (left + x_offset, bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add labels
    cv2.putText(combined_image, "KNOWN IMAGE", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(combined_image, "UNKNOWN IMAGE", (known_width + 30, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the combined image
    window_name = "Face Comparison - Matches Found"
    cv2.imshow(window_name, combined_image)
    print("Face matches found! Press any key to close the comparison window...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def display_no_matches(known_image, unknown_image, known_locations, unknown_locations):
    """
    Display both images side by side with faces highlighted in red (no matches)
    """
    # Get image dimensions
    known_height, known_width = known_image.shape[:2]
    unknown_height, unknown_width = unknown_image.shape[:2]
    
    # Create a combined image (side by side)
    max_height = max(known_height, unknown_height)
    combined_width = known_width + unknown_width + 20  # 20px gap
    
    # Create white background
    combined_image = np.ones((max_height, combined_width, 3), dtype=np.uint8) * 255
    
    # Place images
    combined_image[0:known_height, 0:known_width] = known_image
    combined_image[0:unknown_height, known_width + 20:known_width + 20 + unknown_width] = unknown_image
    
    # Draw red rectangles around faces (no matches)
    for i, (top, right, bottom, left) in enumerate(known_locations):
        cv2.rectangle(combined_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(combined_image, f"NO MATCH K{i+1}", (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    for i, (top, right, bottom, left) in enumerate(unknown_locations):
        x_offset = known_width + 20
        cv2.rectangle(combined_image, (left + x_offset, top), (right + x_offset, bottom), (0, 0, 255), 2)
        cv2.putText(combined_image, f"NO MATCH U{i+1}", (left + x_offset, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add labels
    cv2.putText(combined_image, "KNOWN IMAGE", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(combined_image, "UNKNOWN IMAGE", (known_width + 30, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the combined image
    window_name = "Face Comparison - No Matches"
    cv2.imshow(window_name, combined_image)
    print("No face matches found. Press any key to close the comparison window...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)