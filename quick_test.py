import cv2
import numpy as np
from openpose_caffe import OpenPoseCaffe

def quick_test():
    """Quick test to verify OpenPose is working"""
    
    # Initialize OpenPose
    print("Initializing OpenPose...")
    openpose = OpenPoseCaffe("openpose/models/")
    
    # Create a test image (or load one)
    print("\nCreating test image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple stick figure
    cv2.circle(img, (320, 100), 20, (255, 255, 255), -1)  # Head
    cv2.line(img, (320, 120), (320, 250), (255, 255, 255), 3)  # Body
    cv2.line(img, (320, 150), (250, 200), (255, 255, 255), 3)  # Left arm
    cv2.line(img, (320, 150), (390, 200), (255, 255, 255), 3)  # Right arm
    cv2.line(img, (320, 250), (280, 350), (255, 255, 255), 3)  # Left leg
    cv2.line(img, (320, 250), (360, 350), (255, 255, 255), 3)  # Right leg
    
    # Process with OpenPose
    print("\nProcessing image with OpenPose...")
    keypoints, annotated = openpose.process_image(img)
    
    # Display results
    cv2.imshow('Original', img)
    cv2.imshow('OpenPose Detection', annotated)
    
    print("\nKeypoints found:")
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.1:
            print(f"  Point {i}: ({x:.1f}, {y:.1f}), confidence: {conf:.2f}")
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save results
    cv2.imwrite('test_output.jpg', annotated)
    print("\nOutput saved as 'test_output.jpg'")
    
    # Test with webcam if available
    test_webcam = input("\nTest with webcam? (y/n): ").lower()
    if test_webcam == 'y':
        test_webcam_demo(openpose)

def test_webcam_demo(openpose):
    """Test with webcam"""
    cap = cv2.VideoCapture(0)
    
    print("\nWebcam test - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        keypoints, processed = openpose.process_image(frame)
        
        # Simple fall detection (check if person is horizontal)
        if len(keypoints) >= 25:
            neck = keypoints[1] if keypoints[1][2] > 0.1 else None
            hip = keypoints[8] if keypoints[8][2] > 0.1 else None
            
            if neck and hip:
                dx = abs(neck[0] - hip[0])
                dy = abs(neck[1] - hip[1])
                
                if dy > 0:
                    ratio = dx / dy
                    if ratio > 1.2:
                        cv2.putText(processed, "FALL DETECTED!", 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 0, 255), 2)
        
        cv2.imshow('Webcam - OpenPose', processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    quick_test()