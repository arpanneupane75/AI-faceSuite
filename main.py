
import cv2
import numpy as np
import dlib
import face_recognition
from keras.models import load_model
import os
import time
import sys # For exiting gracefully
from scipy.spatial import distance as dist # For EAR/MAR calculation

# ---------------------------
# Paths
# ---------------------------
deploy_dir = "deploy"
filters_dir = "filters"
outputs_dir = "outputs"
models_dir = "models"
known_faces_dir = "known_faces"
unknown_dir = "unknown_faces"

# Create directories if they don't exist
try:
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(unknown_dir, exist_ok=True)
except OSError as e:
    print(f"Error creating directories: {e}")
    sys.exit(1)

# ---------------------------
# Load Models
# ---------------------------
face_net = None
age_net = None
gender_net = None
emotion_model = None
predictor = None

# Helper function for safe model loading
def load_caffe_model(prototxt_path, caffemodel_path, model_name):
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        if net.empty():
            print(f"Error: Loaded {model_name} network is empty. Check model files.")
            return None
        print(f"Successfully loaded {model_name}.")
        return net
    except cv2.error as e:
        print(f"Error loading {model_name} Caffe model from '{prototxt_path}' and '{caffemodel_path}': {e}")
        return None
    except FileNotFoundError:
        print(f"Error: One or both files for {model_name} not found: '{prototxt_path}', '{caffemodel_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {model_name}: {e}")
        return None

try:
    face_net = load_caffe_model(f"{deploy_dir}/deploy.prototxt", f"{deploy_dir}/res10_300x300_ssd_iter_140000.caffemodel", "Face Detection")
    age_net = load_caffe_model(f"{models_dir}/age_deploy.prototxt", f"{models_dir}/age_net.caffemodel", "Age Estimation")
    gender_net = load_caffe_model(f"{models_dir}/gender_deploy.prototxt", f"{models_dir}/gender_net.caffemodel", "Gender Estimation")

    emotion_model_path = f"{models_dir}/fer2013_mini_XCEPTION.102-0.66.hdf5"
    if os.path.exists(emotion_model_path):
        emotion_model = load_model(emotion_model_path, compile=False)
        print("Successfully loaded Emotion Model.")
    else:
        print(f"Error: Emotion model not found at '{emotion_model_path}'.")

    predictor_path = f"{models_dir}/shape_predictor_68_face_landmarks.dat"
    if os.path.exists(predictor_path):
        predictor = dlib.shape_predictor(predictor_path)
        print("Successfully loaded Dlib Shape Predictor.")
    else:
        print(f"Error: Dlib shape predictor not found at '{predictor_path}'.")

except Exception as e:
    print(f"An error occurred during model loading: {e}")
    sys.exit(1)

# Exit if critical models are not loaded
if not (face_net and age_net and gender_net and emotion_model and predictor):
    print("One or more critical models failed to load. Exiting.")
    sys.exit(1)

AGE_LIST = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
GENDER_LIST = ['Male','Female']
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ---------------------------
# Drowsiness and Smiling Detection Constants
# ---------------------------
EYE_AR_THRESH = 0.25 # Threshold for eye aspect ratio to indicate a blink
EYE_AR_CONSEC_FRAMES = 10 # Number of consecutive frames the eye must be below the threshold
MOUTH_AR_THRESH = 0.50 # Threshold for mouth aspect ratio to indicate a yawn
MOUTH_AR_CONSEC_FRAMES = 10 # Number of consecutive frames the mouth must be above the threshold
SMILING_THRESH = 0.35 # Threshold for smiling detection (adjust empirically)

# Facial landmark indices for eyes and mouth (dlib's 68-point model)
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42
LEFT_EYE_START, LEFT_EYE_END = 42, 48
MOUTH_OUTER_START, MOUTH_OUTER_END = 48, 60 # Outer mouth points for smile detection
MOUTH_INNER_START, MOUTH_INNER_END = 60, 68 # Inner mouth points for yawn detection

# ---------------------------
# Load Known Faces
# ---------------------------
known_face_encodings = []
known_face_names = []

if not os.path.exists(known_faces_dir) or not os.listdir(known_faces_dir):
    print(f"Warning: No known faces found in '{known_faces_dir}'. All faces will be marked as 'Unknown'.")
else:
    for filename in os.listdir(known_faces_dir):
        if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            continue # Skip non-image files

        full_path = f"{known_faces_dir}/{filename}"
        try:
            img = face_recognition.load_image_file(full_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"Warning: No face found in '{filename}'. Skipping for known faces.")
        except Exception as e:
            print(f"Error loading or processing known face '{filename}': {e}")
    print(f"Loaded {len(known_face_encodings)} known faces.")

# ---------------------------
# Filters and Colors
# ---------------------------
filters = {}
filter_paths = {
    "Happy": f"{filters_dir}/sunglasses.png",
    "Sad": f"{filters_dir}/hat.png",
    "Angry": f"{filters_dir}/angry_emoji.png"
}

for emotion, path in filter_paths.items():
    try:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                filters[emotion] = img
            else:
                print(f"Warning: Could not read filter image '{path}'. It might be corrupted or an invalid image format.")
        else:
            print(f"Warning: Filter image not found at '{path}'. Skipping this filter.")
    except Exception as e:
        print(f"Error loading filter '{path}': {e}")

emotion_colors = {
    "Happy": (0,255,0),
    "Sad": (255,0,0),
    "Angry": (0,0,255),
    "Neutral": (200,200,200),
    "Surprise": (0,255,255),
    "Fear": (255,0,255),
    "Disgust": (0,128,0)
}

# ---------------------------
# Helper Functions
# ---------------------------
def overlay_image(background, overlay, x, y, w, h):
    if overlay is None or background is None or w <= 0 or h <= 0:
        return background

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(background.shape[1], x + w), min(background.shape[0], y + h)
    
    effective_w = x2 - x1
    effective_h = y2 - y1

    if effective_w <= 0 or effective_h <= 0:
        return background

    try:
        overlay_resized = cv2.resize(overlay, (effective_w, effective_h))
    except cv2.error as e:
        # print(f"Error resizing overlay image: {e}") # Suppress frequent errors
        return background

    if overlay_resized.shape[2] == 4: # RGBA
        alpha = overlay_resized[:,:,3] / 255.0
        for c in range(3):
            background[y1:y2, x1:x2, c] = (alpha * overlay_resized[:,:,c] +
                                           (1-alpha) * background[y1:y2, x1:x2, c])
    else: # RGB or Grayscale, convert to 3 channels if needed
        if len(overlay_resized.shape) == 2:
            overlay_resized = cv2.cvtColor(overlay_resized, cv2.COLOR_GRAY2BGR)
        background[y1:y2, x1:x2] = overlay_resized
    return background

def get_landmarks(face_rect, gray_frame):
    if predictor is None:
        return None
    try:
        # dlib.rectangle expects (left, top, right, bottom)
        dlib_rect = dlib.rectangle(face_rect[0], face_rect[1], face_rect[2], face_rect[3])
        shape = predictor(gray_frame, dlib_rect)
        
        coords = np.zeros((68,2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    except Exception as e:
        # print(f"Error getting landmarks: {e}") 
        return None

def cartoonify(face_img):
    if face_img is None or face_img.size == 0:
        return face_img
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,9,2)
        color = cv2.bilateralFilter(face_img,9,250,250)
        cartoon = cv2.bitwise_and(color,color,mask=edges)
        return cartoon
    except Exception as e:
        # print(f"Error in cartoonify: {e}")
        return face_img 

def aging_filter(face_img):
    if face_img is None or face_img.size == 0:
        return face_img
    try:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]) # Sharpening filter
        aged = cv2.filter2D(face_img, -1, kernel)
        return cv2.GaussianBlur(aged,(7,7),0)
    except Exception as e:
        # print(f"Error in aging_filter: {e}")
        return face_img 

def head_pose_estimation(landmarks, frame_size):
    if landmarks is None or len(landmarks) < 68: # Need all 68 landmarks
        return None, None
    
    # Model points are 3D coordinates of key facial features
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (point 30)
        (0.0, -330.0, -65.0),        # Chin (point 8)
        (-225.0, 170.0, -135.0),     # Left eye left corner (point 36)
        (225.0, 170.0, -135.0),      # Right eye right corner (point 45)
        (-150.0, -150.0, -125.0),    # Left mouth corner (point 48)
        (150.0, -150.0, -125.0)      # Right mouth corner (point 54)
    ])
    
    # Image points are 2D coordinates of the same features in the image
    image_points = np.array([
        landmarks[30], landmarks[8], landmarks[36], landmarks[45], landmarks[48], landmarks[54]
    ], dtype="double")
    
    # Camera internals
    focal_length = frame_size[1] 
    center = (frame_size[1]//2, frame_size[0]//2)
    camera_matrix = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]], dtype="double")
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    
    try:
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            return rvec, tvec
        else:
            return None, None
    except cv2.error as e:
        return None, None
    except Exception as e:
        return None, None

def draw_axes(frame, rvec, tvec, center, camera_matrix):
    if rvec is None or tvec is None or camera_matrix is None:
        return
    axis = np.float32([[50,0,0],[0,50,0],[0,0,50]]).reshape(-1,3)
    try:
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, np.zeros((4,1)))
        corner = tuple(center)
        imgpts = imgpts.astype(int)
        
        # Draw axes only if start and end points are valid and within frame
        if 0 <= corner[0] < frame.shape[1] and 0 <= corner[1] < frame.shape[0]:
            if imgpts.shape[0] >= 3:
                cv2.line(frame, corner, tuple(imgpts[0].ravel()), (0,0,255),2) # X-red
                cv2.line(frame, corner, tuple(imgpts[1].ravel()), (0,255,0),2) # Y-green
                cv2.line(frame, corner, tuple(imgpts[2].ravel()), (255,0,0),2) # Z-blue
    except Exception as e:
        pass 

# Drowsiness and Smiling Detection Helper Functions
def eye_aspect_ratio(eye):
    # compute the Euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the Euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth): # For Yawning
    # compute the Euclidean distances between the three sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[1], mouth[7]) # 61, 67
    B = dist.euclidean(mouth[2], mouth[6]) # 62, 66
    C = dist.euclidean(mouth[3], mouth[5]) # 63, 65
    # compute the Euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0], mouth[4]) # 60, 64
    # compute the mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)
    return mar

def smile_ratio(mouth): # For Smiling
    # Distance between outer corners of the mouth
    A = dist.euclidean(mouth[0], mouth[6]) # 48, 54 (horizontal)
    # Average distance between upper and lower lip points
    B = dist.euclidean(mouth[2], mouth[10]) # 50, 58
    C = dist.euclidean(mouth[4], mouth[8]) # 52, 56
    
    # Avoid division by zero if mouth is perfectly flat horizontally
    if A == 0:
        return 0.0
    
    # Ratio of vertical openness to horizontal width
    # A larger ratio typically indicates a wider, more open mouth for a smile
    # This might need tuning based on different face types
    sm_ratio = (B + C) / (2.0 * A) 
    return sm_ratio


# ---------------------------
# Tracker & Face Data
# ---------------------------
tracker_data = {}
face_counter = 0
FRAME_SKIP = 5 # Process age/gender/emotion/name every 5 frames for efficiency
emotion_counts = {e:0 for e in emotion_labels}

# Performance metrics
frame_processing_times = []
detection_times = []
analysis_times = []

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Check if camera is connected or in use.")
    sys.exit(1)

prev_time = time.time()
unknown_face_counter = 0

print("Press 'q' to quit")

while True:
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera. Exiting...")
        break
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_frame_fids = set() # To track which FIDs are active in this frame

    new_tracker_data = {}
    # Update existing trackers and process faces
    for fid, data in list(tracker_data.items()):
        try:
            success, box = data['tracker'].update(frame)
        except Exception as e:
            # print(f"Tracker {fid} update failed: {e}. Removing tracker.") # Suppress frequent warnings
            success = False # Force failure if tracker throws error

        if success:
            x,y,bw,bh = [int(v) for v in box]

            # Ensure bounding box is within frame boundaries and valid
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            bw, bh = x2 - x, y2 - y

            if bw <= 0 or bh <= 0:
                continue

            face_roi = frame[y:y2, x:x2]
            if face_roi.size == 0:
                continue
            
            # --- Advanced Feature: Drowsiness & Smiling Detection ---
            landmarks = get_landmarks((x, y, x2, y2), gray)
            data['landmarks'] = landmarks # Store landmarks for persistent use within tracker_data

            if landmarks is not None:
                # Eye Aspect Ratio for Drowsiness
                left_eye = landmarks[LEFT_EYE_START:LEFT_EYE_END]
                right_eye = landmarks[RIGHT_EYE_START:RIGHT_EYE_END]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    data['ear_consec_frames'] += 1
                else:
                    data['ear_consec_frames'] = 0
                
                data['is_drowsy'] = data['ear_consec_frames'] >= EYE_AR_CONSEC_FRAMES

                # Mouth Aspect Ratio for Yawning
                mouth_inner = landmarks[MOUTH_INNER_START:MOUTH_INNER_END]
                mar = mouth_aspect_ratio(mouth_inner)

                if mar > MOUTH_AR_THRESH:
                    data['mar_consec_frames'] += 1
                else:
                    data['mar_consec_frames'] = 0
                
                data['is_yawning'] = data['mar_consec_frames'] >= MOUTH_AR_CONSEC_FRAMES

                # Smile Ratio for Smiling
                mouth_outer = landmarks[MOUTH_OUTER_START:MOUTH_OUTER_END]
                sm_ratio_val = smile_ratio(mouth_outer)
                data['is_smiling'] = sm_ratio_val > SMILING_THRESH
            else:
                data['ear_consec_frames'] = 0
                data['mar_consec_frames'] = 0
                data['is_drowsy'] = False
                data['is_yawning'] = False
                data['is_smiling'] = False

            # --- Periodic Analysis (Emotion, Age, Gender, Face Recognition) ---
            if data['frame_count'] % FRAME_SKIP == 0:
                analysis_start_time = time.time()
                try:
                    # Emotion prediction & Confidence
                    gray_face_resized = cv2.resize(gray[y:y2, x:x2], (64,64)).reshape(1,64,64,1)/255.0
                    emotion_pred = emotion_model.predict(gray_face_resized, verbose=0)
                    data['emotion'] = emotion_labels[np.argmax(emotion_pred)]
                    data['emotion_conf'] = np.max(emotion_pred) * 100 # Percentage

                    # Age and Gender prediction & Confidence
                    blob_dnn = cv2.dnn.blobFromImage(face_roi,1.0,(227,227),(78.426,87.768,114.895),swapRB=False)
                    gender_net.setInput(blob_dnn)
                    gender_preds = gender_net.forward()
                    if gender_preds.size > 0:
                        data['gender'] = GENDER_LIST[gender_preds[0].argmax()]
                        data['gender_conf'] = np.max(gender_preds) * 100 # Percentage
                    else:
                        data['gender'] = "Unknown" 
                        data['gender_conf'] = 0.0

                    age_net.setInput(blob_dnn)
                    age_preds = age_net.forward()
                    if age_preds.size > 0:
                        data['age'] = AGE_LIST[age_preds[0].argmax()]
                        data['age_conf'] = np.max(age_preds) * 100 # Percentage
                    else:
                        data['age'] = "Unknown" 
                        data['age_conf'] = 0.0

                    # Face Recognition (and storing encoding for re-identification)
                    # Convert ROI to RGB for face_recognition library
                    face_rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    # Use dlib.rectangle to define the ROI for face_recognition.face_encodings
                    # Note: face_recognition expects (top, right, bottom, left) order
                    # For an ROI, it's typically (0, width, height, 0)
                    face_locations_roi = [(0, face_rgb_roi.shape[1], face_rgb_roi.shape[0], 0)]
                    
                    face_encodings_in_roi = face_recognition.face_encodings(face_rgb_roi, 
                                                                             known_face_locations=face_locations_roi, 
                                                                             num_jitters=1)
                    if face_encodings_in_roi:
                        data['face_encoding'] = face_encodings_in_roi[0] # Store the encoding
                        matches = face_recognition.compare_faces(known_face_encodings, data['face_encoding'])
                        name = "Unknown"
                        if True in matches:
                            name = known_face_names[matches.index(True)]
                        else:
                            try:
                                unknown_face_counter += 1
                                save_path = f"{unknown_dir}/unknown_{unknown_face_counter}.jpg"
                                cv2.imwrite(save_path, face_roi)
                            except Exception as save_err:
                                print(f"Error saving unknown face: {save_err}")
                        data['name'] = name
                    else:
                        data['name'] = "Unknown" 
                        data['face_encoding'] = None # No encoding found

                except Exception as e:
                    # print(f"Error during periodic analysis for tracker {fid}: {e}") # Suppress frequent errors
                    pass
                analysis_times.append(time.time() - analysis_start_time)


            # --- Drawing & Visuals ---
            if landmarks is not None:
                center_point = ((x+x2)//2,(y+y2)//2)
                focal_length_draw = w 
                center_draw = (w//2, h//2)
                camera_matrix_draw = np.array([[focal_length_draw,0,center_draw[0]],
                                            [0,focal_length_draw,center_draw[1]],
                                            [0,0,1]], dtype="double")
                rvec, tvec = head_pose_estimation(landmarks, frame.shape)
                draw_axes(frame, rvec, tvec, center_point, camera_matrix_draw)

                # Draw eye and mouth landmarks for debugging drowsiness/smiling
                # for (start, end) in [(RIGHT_EYE_START, RIGHT_EYE_END), (LEFT_EYE_START, LEFT_EYE_END), 
                #                       (MOUTH_OUTER_START, MOUTH_OUTER_END), (MOUTH_INNER_START, MOUTH_INNER_END)]:
                #     for i in range(start, end):
                #         cv2.circle(frame, (landmarks[i][0], landmarks[i][1]), 1, (0, 255, 255), -1)


            if data['emotion'] in filters and filters[data['emotion']] is not None:
                filter_y_offset = int(bh * 0.2) 
                filter_h = int(bh * 0.6)
                filter_w = bw
                overlay_y = y - filter_y_offset
                if overlay_y < 0: overlay_y = 0
                overlay_image(frame, filters[data['emotion']], x, overlay_y, filter_w, filter_h)

            if data['emotion']=="Happy":
                try: frame[y:y2, x:x2] = cartoonify(face_roi)
                except Exception as e: pass
            elif data['emotion']=="Sad":
                try: frame[y:y2, x:x2] = aging_filter(face_roi)
                except Exception as e: pass

            color = emotion_colors.get(data['emotion'],(0,255,0))
            cv2.rectangle(frame,(x,y),(x2,y2),color,2)
            
            # Ensure text position is within frame boundaries
            text_offset_y = 0
            # Name
            name_text = f"{data.get('name','')}"
            cv2.putText(frame, name_text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # ID and Emotion with Confidence
            emotion_text = f"ID:{fid} {data['emotion']}"
            if 'emotion_conf' in data and data['emotion_conf'] > 0:
                emotion_text += f" ({data['emotion_conf']:.1f}%)"
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Gender and Age with Confidence
            gender_age_text = f"{data['gender']}"
            if 'gender_conf' in data and data['gender_conf'] > 0:
                gender_age_text += f" ({data['gender_conf']:.1f}%)"
            gender_age_text += f", {data['age']}"
            if 'age_conf' in data and data['age_conf'] > 0:
                gender_age_text += f" ({data['age_conf']:.1f}%)"
            
            cv2.putText(frame, gender_age_text, (x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # Display Drowsiness/Yawn/Smile status
            status_y_pos = y2 + 45
            if data['is_drowsy']:
                cv2.putText(frame,"ALERT: DROWSY!",(x, status_y_pos),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                status_y_pos += 25
            if data['is_yawning']:
                cv2.putText(frame,"YAWNING!",(x, status_y_pos),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,165,255),2)
                status_y_pos += 25
            if data['is_smiling']:
                cv2.putText(frame,"SMILING!",(x, status_y_pos),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                status_y_pos += 25


            data['last_position'] = (x,y,x2,y2)
            data['frame_count'] +=1
            new_tracker_data[fid] = data
            current_frame_fids.add(fid) # Mark this tracker as active in the current frame

    tracker_data = new_tracker_data

    # Detect new faces and initialize trackers for them
    detection_start_time = time.time()
    try:
        blob_detection = cv2.dnn.blobFromImage(frame,1.0,(300,300),[104,117,123],False,False)
        face_net.setInput(blob_detection)
        detections = face_net.forward()
    except Exception as e:
        print(f"Error during face detection DNN forward pass: {e}")
        detections = np.array([]) # Ensure detections is an empty numpy array if error occurs
    detection_times.append(time.time() - detection_start_time)

    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf>0.6: # Confidence threshold for detection
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)

            # Ensure detected box is valid and within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            matched_existing_tracker = False
            for fid,d in tracker_data.items():
                tx1,ty1,tx2,ty2 = d['last_position']
                
                # Calculate Intersection over Union (IoU) for robust matching
                inter_x1 = max(x1, tx1)
                inter_y1 = max(y1, ty1)
                inter_x2 = min(x2, tx2)
                inter_y2 = min(y2, ty2)

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (tx2 - tx1) * (ty2 - ty1)
                union_area = float(box1_area + box2_area - inter_area)

                if union_area > 0 and (inter_area / union_area) > 0.5: # IoU threshold for overlap
                    matched_existing_tracker = True
                    # If this detection significantly overlaps with an existing tracker,
                    # and that tracker wasn't updated successfully this frame, re-init it.
                    if fid not in current_frame_fids: 
                        try:
                            tracker_data[fid]['tracker'] = cv2.legacy.TrackerCSRT_create()
                            tracker_data[fid]['tracker'].init(frame,(x1,y1,x2-x1,y2-y1))
                            tracker_data[fid]['last_position'] = (x1,y1,x2,y2)
                            tracker_data[fid]['frame_count'] = 0 # Reset frame count to re-analyze soon
                            current_frame_fids.add(fid) # Mark it as now active
                        except Exception as e:
                            print(f"Error re-initializing tracker {fid}: {e}")
                    break # Matched, so this detection is handled

            if not matched_existing_tracker:
                try:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    if (x2-x1) > 0 and (y2-y1) > 0:
                        tracker.init(frame,(x1,y1,x2-x1,y2-y1))
                        tracker_data[str(face_counter)] = {
                            'tracker': tracker,
                            'last_position': (x1,y1,x2,y2),
                            'age': '', 'gender': '', 'emotion': '',
                            'age_conf': 0.0, 'gender_conf': 0.0, 'emotion_conf': 0.0,
                            'frame_count': 0,
                            'name': 'Unknown',
                            'face_encoding': None, # Store encoding here
                            'ear_consec_frames': 0, 'mar_consec_frames': 0,
                            'is_drowsy': False, 'is_yawning': False, 'is_smiling': False,
                            'landmarks': None # Store landmarks here for persistence
                        }
                        face_counter +=1
                except Exception as e:
                    print(f"Error initializing new tracker: {e}")

    # Emotion heatmap
    heatmap_overlay = np.zeros_like(frame, dtype=np.uint8)
    # Reset counts for the current frame based on active trackers
    for emo in emotion_counts:
        emotion_counts[emo] = 0
    
    for fid,data in tracker_data.items():
        if fid in current_frame_fids: # Only count for currently active faces
            x1,y1,x2,y2 = data['last_position']
            color = emotion_colors.get(data['emotion'],(0,255,0))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(heatmap_overlay,(x1,y1),(x2,y2),color,-1) 
                if data['emotion']:
                    emotion_counts[data['emotion']] += 1 # Recalculate based on currently visible faces

    frame = cv2.addWeighted(frame,0.7,heatmap_overlay,0.3,0)

    # Analytics display (top-left corner)
    stats_rect_width = 300
    stats_rect_height = 250 # Increased height for more stats
    stats_rect_x2 = min(stats_rect_width, w)
    stats_rect_y2 = min(stats_rect_height, h) 
    cv2.rectangle(frame,(0,0),(stats_rect_x2,stats_rect_y2),(0,0,0),-1) # Background for text
    cv2.addWeighted(frame,1,frame,0.4,0,frame) # Blend effect

    # Display Tracked Faces Count
    cv2.putText(frame,f"Tracked Faces: {len(current_frame_fids)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    # Display Emotion Counts
    y_off_emotions = 60
    cv2.putText(frame, "Current Emotions:", (10, y_off_emotions), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    y_off_emotions += 20
    for emo,count in emotion_counts.items():
        if count > 0 and y_off_emotions < stats_rect_y2 - 10: 
            cv2.putText(frame,f"  {emo}: {count}",(10,y_off_emotions),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            y_off_emotions+=20
        elif y_off_emotions >= stats_rect_y2 - 10:
            break 

    # FPS calculation and display (top-right corner)
    current_time = time.time()
    delta_time = current_time - prev_time
    if delta_time > 0: 
        fps = 1 / delta_time
    else:
        fps = 0 
    prev_time = current_time
    frame_processing_times.append(time.time() - frame_start_time)

    # Average times (last 30 frames for smoother display)
    avg_frame_time = np.mean(frame_processing_times[-30:]) * 1000 if frame_processing_times else 0
    avg_det_time = np.mean(detection_times[-30:]) * 1000 if detection_times else 0
    avg_analysis_time = np.mean(analysis_times[-30:]) * 1000 if analysis_times else 0

    fps_stats_x = frame.shape[1] - (stats_rect_width - 50) # Position from right edge
    if fps_stats_x < stats_rect_x2 + 10: # Ensure it doesn't overlap with left panel
        fps_stats_x = frame.shape[1] - stats_rect_width - 10 # Shift left if overlap

    # Background for FPS/performance stats
    fps_stats_rect_x1 = max(0, fps_stats_x - 10) # Padding
    fps_stats_rect_y1 = 0
    fps_stats_rect_x2 = frame.shape[1]
    fps_stats_rect_y2 = min(150, h) # Fixed height for FPS stats
    cv2.rectangle(frame, (fps_stats_rect_x1, fps_stats_rect_y1), (fps_stats_rect_x2, fps_stats_rect_y2), (0,0,0), -1)
    cv2.addWeighted(frame,1,frame,0.4,0,frame) # Blend effect

    cv2.putText(frame,f"FPS: {int(fps)}",(fps_stats_x,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv2.putText(frame,f"Frame Proc: {avg_frame_time:.1f} ms",(fps_stats_x,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(frame,f"Detection: {avg_det_time:.1f} ms",(fps_stats_x,85),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(frame,f"Analysis (Avg/Face): {avg_analysis_time:.1f} ms",(fps_stats_x,110),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)


    try:
        cv2.imshow("Ultimate Face AI Suite", frame)
    except cv2.error as e:
        print(f"Error displaying frame: {e}. Window might be closed.")
        break
    except Exception as e:
        print(f"An unexpected error occurred during display: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()