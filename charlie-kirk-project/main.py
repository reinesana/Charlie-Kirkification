import os
import sys
import time
from pathlib import Path

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
# Remove outliers using mean and standard deviation
import numpy as np
import pygame  # pyright: ignore[reportMissingImports]
from PIL import Image

# Initialize pygame mixer
pygame.mixer.init()

images_ref = []

# Check if audio file exists
audio_file = "./assets/we-are-charlie-kirk-song.mp3"
if not os.path.exists(audio_file):
    print(f"Error: Audio file not found: {audio_file}")
    sys.exit(1)

pygame.mixer.music.load(audio_file)


face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
spam = Path("assets/spam")

# Check if spam directory exists
if not spam.exists():
    print(f"Error: Spam directory not found: {spam}")
    sys.exit(1)

# Check if there are images in spam directory
spam_images = list(spam.glob("*.jpg")) + list(spam.glob("*.png"))
if not spam_images:
    print(f"Error: No images found in {spam}")
    sys.exit(1)

timer = 2.0
timer_started = None
playing = False

# List to track opened images (though PIL can't actually close external viewer windows)
images_ref = []

# Calibration variables
calibrating = True
calibration_samples = []
calibration_count = 0
initial_calibration_position= None
baseline_l_ratio = None
baseline_r_ratio = None
look_down_threshold = 0.15  # How much below baseline to trigger
movement_threshold = 0.1  # Movement detection threshold during calibration

# Movement detection for calibration reset
dataset_consistency_threshold = 0.05  # Max standard deviation allowed in final dataset
outlier_std_threshold = 2.0  # Remove samples beyond 2 standard deviations
calibration_start_time = None
countdown_phase = True
countdown_start = None
blink_tolerance_frames = 3  # Allow a few frames of missing/bad data for blinks
bad_frame_count = 0

# Initialize camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera")
    sys.exit(1)

print("=== CHARLIE KIRK DETECTION SYSTEM ===")
print("Starting calibration process...")
print("Instructions:")
print("1. Position yourself comfortably in front of the camera")
print("2. Look straight ahead at the camera")
print("3. Stay still during the 3-second countdown and calibration")
print("4. Press ESC to quit anytime")
print("Get ready...")


while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    height, width, depth = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_frame)
    face_landmark_points = processed_image.multi_face_landmarks

    if face_landmark_points:
        bad_frame_count = 0  # Reset bad frame count when face is detected
        one_face_landmark_points = face_landmark_points[0].landmark
        
        left = [one_face_landmark_points[145], one_face_landmark_points[159]]
        for landmark_point in left:
            x = int(landmark_point.x * width)
            y = int(landmark_point.y * height)
            #print(x, y)
            cv2.circle(frame, (x,y), 3, (0,255,255))


        right = [one_face_landmark_points[374], one_face_landmark_points[386]]
        for landmark_point in right:
                x = int(landmark_point.x * width)
                y = int(landmark_point.y * height)
                #print(x, y)
                cv2.circle(frame, (x,y), 3, (255,255,0))

        l_iris = one_face_landmark_points[468]
        r_iris = one_face_landmark_points[473]
        
        l_ratio = (l_iris.y  - left[1].y)  / (left[0].y  - left[1].y  + 1e-6)
        r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)

        if calibrating:
            current_time = time.time()
            
            if countdown_phase:
                # Initialize countdown
                if countdown_start is None:
                    countdown_start = current_time
                
                # Calculate countdown
                elapsed = current_time - countdown_start
                countdown_remaining = max(0, 3 - elapsed)
                
                if countdown_remaining > 0:
                    # Show countdown
                    countdown_text = f"Get ready: {int(countdown_remaining) + 1}"
                    cv2.putText(frame, countdown_text, (width//2 - 150, height//2 - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, "Look straight at the camera", (width//2 - 150, height//2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, "Stay still", (width//2 - 150, height//2 + 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                else:
                    # Countdown finished, start calibration
                    countdown_phase = False
                    calibration_start_time = current_time
                    calibration_samples = []
                    calibration_count = 0
                    print("Calibration started!")
            
            else:
                # Calibration phase
                calibration_samples.append((l_ratio, r_ratio))
                calibration_count += 1
                
                # Show calibration progress
                progress = (calibration_count / 90) * 100
                cv2.putText(frame, f"CALIBRATING: {progress:.0f}%", (width//2 - 150, height//2 - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, "Keep looking straight!", (width//2 - 130, height//2 - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Blinking is OK", (width//2 - 80, height//2 + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Samples: {calibration_count}/90", (width//2 - 90, height//2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw progress bar
                bar_width = 300
                bar_height = 20
                bar_x = width//2 - bar_width//2
                bar_y = height//2 + 70
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                fill_width = int((progress / 100) * bar_width)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 255), -1)
                
                # Collect 90 samples (about 3 seconds at ~30fps)
                if calibration_count >= 90:
                    # Calculate baseline from collected samples with outlier detection
                    l_ratios = [sample[0] for sample in calibration_samples]
                    r_ratios = [sample[1] for sample in calibration_samples]
                    


                    # Left eye outlier removal
                    l_mean = np.mean(l_ratios)
                    l_std = np.std(l_ratios)
                    l_filtered = [x for x in l_ratios if abs(x - l_mean) <= outlier_std_threshold * l_std]
                    
                    # Right eye outlier removal
                    r_mean = np.mean(r_ratios)
                    r_std = np.std(r_ratios)
                    r_filtered = [x for x in r_ratios if abs(x - r_mean) <= outlier_std_threshold * r_std]
                    
                    # Calculate final baselines from filtered data
                    baseline_l_ratio = np.mean(l_filtered) if l_filtered else l_mean
                    baseline_r_ratio = np.mean(r_filtered) if r_filtered else r_mean
                    
                    # Calculate final standard deviations for quality check
                    final_l_std = np.std(l_filtered) if l_filtered else l_std
                    final_r_std = np.std(r_filtered) if r_filtered else r_std
                    
                    outliers_removed = len(calibration_samples) - len(l_filtered) - len(r_filtered)
                    
                    # Check dataset consistency - if too inconsistent, restart calibration
                    max_std = max(final_l_std, final_r_std)
                    if max_std > dataset_consistency_threshold:
                        print(f"Dataset inconsistent! Max std: {max_std:.3f} > threshold: {dataset_consistency_threshold}")
                        print("Too much movement detected during calibration. Restarting...")
                        # Reset calibration
                        countdown_phase = True
                        countdown_start = None
                        calibration_samples = []
                        calibration_count = 0
                        initial_calibration_position = None
                        cv2.putText(frame, "INCONSISTENT DATA!", (width//2 - 150, height//2 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, "Too much movement - restarting...", (width//2 - 180, height//2 + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow('Face Detection', frame)
                        cv2.waitKey(2000)  # Show message for 2 seconds
                        continue
                    
                    calibrating = False
                    print(f"Calibration complete! Outliers removed: {outliers_removed}")
                    print(f"Baselines: L={baseline_l_ratio:.3f}±{final_l_std:.3f}, R={baseline_r_ratio:.3f}±{final_r_std:.3f}")
                    print(f"Samples used: L={len(l_filtered)}/{len(l_ratios)}, R={len(r_filtered)}/{len(r_ratios)}")
                    print("Now monitoring for downward gaze...")
                    
                    # Show completion message
                    cv2.putText(frame, "CALIBRATION COMPLETE!", (width//2 - 180, height//2 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "System is now active", (width//2 - 130, height//2 + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('Face Detection', frame)
                    cv2.waitKey(2000)  # Show message for 2 seconds
                    
        else:
            # Normal detection phase
            current = time.time()
            
            # Calculate how much below baseline the current ratios are
            l_below_baseline = baseline_l_ratio - l_ratio if baseline_l_ratio else 0
            r_below_baseline = baseline_r_ratio - r_ratio if baseline_r_ratio else 0
            
            # Show current status with better formatting
            cv2.putText(frame, "SYSTEM ACTIVE", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze deviation - L: {l_below_baseline:.2f} R: {r_below_baseline:.2f}", (50, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Check for significant movement that might require recalibration
            if (abs(l_below_baseline) > 0.3 or abs(r_below_baseline) > 0.3):
                cv2.putText(frame, "Large movement detected", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                cv2.putText(frame, "Press 'R' to recalibrate", (50, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            
            # Handle asymmetric eye movement - use maximum deviation
            max_below_baseline = max(l_below_baseline, r_below_baseline)
            
            #print(f'iris ratios - L:{l_ratio:.3f} R:{r_ratio:.3f} | below baseline - L:{l_below_baseline:.3f} R:{r_below_baseline:.3f} | max:{max_below_baseline:.3f}')
            
            # Use maximum deviation - trust the eye that shows strongest downward gaze
            if max_below_baseline > look_down_threshold:
                cv2.putText(frame, "LOOKING DOWN DETECTED!", (50, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if timer_started is None:
                     timer_started = current
                remaining_time = max(0, timer - (current - timer_started))
                cv2.putText(frame, f"Trigger in: {remaining_time:.1f}s", (50, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                if (current - timer_started) >= timer:
                    if not playing:
                        cv2.putText(frame, "CHARLIE KIRK ACTIVATED!", (50, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        pygame.mixer.music.play()
                        for loop in range(4):
                            for i in spam.iterdir():
                                im = Image.open(i)
                                im.show()
                                images_ref.append(im)
                        playing = True
                        
            else: 
                 timer_started = None
                 playing = False
                    

    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(100)

    if key == 27:  # ESC key
        break
    elif key == ord('r') or key == ord('R'):  # R key to recalibrate
        if not calibrating:
            print("Restarting calibration...")
            calibrating = True
            countdown_phase = True
            countdown_start = None
            calibration_samples = []
            calibration_count = 0
            baseline_l_ratio = None
            baseline_r_ratio = None
            initial_calibration_position = None

cam.release()
cv2.destroyAllWindows()
