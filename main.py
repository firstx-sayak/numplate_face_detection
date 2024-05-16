import cv2
import numpy as np
import os
import face_recognition
import pytesseract
from ultralytics import YOLO

# Initialize YOLO model
license_plate_detector = YOLO('best.pt')

# Path to the folder containing face images (database)
database_folder = "DB"

# Sample dictionary with image filenames and associated numbers
face_dict = {
    "<imagename.jpg>": "<platenumber>",
    # Add more images and associated numbers as needed
}

# Function to load face encodings from images in the database folder
def load_database_faces(database_folder):
    known_face_encodings = []
    known_face_numbers = []
    for filename, face_number in face_dict.items():
        image_path = os.path.join(database_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        if len(face_encoding) > 0:
            known_face_encodings.append(face_encoding[0])
            known_face_numbers.append(face_number)
        else:
            print("No face found in", filename)
    return known_face_encodings, known_face_numbers

# Load known face encodings and numbers from images in the database folder
known_face_encodings, known_face_numbers = load_database_faces(database_folder)

# Function to perform face verification and print associated face number
def verify_face(frame):
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return None, None

    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        for i, match in enumerate(matches):
            if match:
                print("Face verified with number:", known_face_numbers[i])
                return known_face_numbers[i], face_location

    return None, face_locations[0]  # Return face location if not verified

# Load video
cap = cv2.VideoCapture(0)

# Minimum confidence score for displaying OCR text
min_confidence = 0.88

best_ocr_value = ""
best_verified_face_key = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from camera")
        break

    # Perform license plate detection
    results_list = license_plate_detector(frame)

    # Iterate over each result in the list
    for results in results_list:
        # Check if license plates are detected
        if results.boxes is not None:
            for license_plate, confidence in zip(results.boxes.xyxy, results.boxes.conf):
                # Extract coordinates and confidence
                x1, y1, x2, y2 = map(int, license_plate[:4])

                # Draw bounding box with blue color for license plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Crop license plate region
                license_plate_crop = frame[y1:y2, x1:x2]

                # Convert to grayscale
                gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                # Perform OCR on license plate region
                text = pytesseract.image_to_string(gray, config='--psm 6')

                # Display OCR text above bounding box with blue color
                cv2.putText(frame, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Keep track of the best OCR value
                if confidence > min_confidence and len(text) > len(best_ocr_value):
                    best_ocr_value = text
                    print("NumberPlate:", best_ocr_value)

    # Perform face detection
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Iterate over each face found
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Perform face verification
        face_number, _ = verify_face(frame)
        print("Face Number:", face_number)
        print("Face Location:", face_location)

        # Draw bounding box for detected face
        top, right, bottom, left = face_location
        if face_number:
            # Draw bounding box with green color
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Display face number (key value) above bounding box in green color
            cv2.putText(frame, f"{face_number}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the OCR value matches the verified face key
            if best_ocr_value and (
                best_ocr_value in str(face_number) or 
                str(face_number) in best_ocr_value
            ):
                print("OCR value:", best_ocr_value)
                print("Face number:", face_number)
                common_letters = set(best_ocr_value).intersection(set(str(face_number)))
                print("Common letters:", common_letters)
                if len(common_letters) >= 3:
                    print("OWNER VERIFIED")
                    # Display "User Verified" on top of face bounding box
                    cv2.putText(frame, "USER VERIFIED", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Draw bounding box with red color
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display frame
    cv2.imshow('License Plate and Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
