import os
import cv2

base_path = r"C:\Users\dhruv\.cache\kagglehub\datasets\gti-upm\leapgestrecog\versions\1\leapGestRecog"

# List user folders
user_dirs = [d for d in os.listdir(base_path) if d.isdigit()]
print("User folders found:", user_dirs)

# Try a specific user and gesture
example_user = "00"
example_gesture = "01_palm"
example_file = "frame_00_01_0001.png"

img_path = os.path.join(base_path, example_user, example_gesture, example_file)
print("🔍 Image path:", img_path)

# Check if file exists
if not os.path.exists(img_path):
    print("❌ File does not exist.")
else:
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image could not be loaded (corrupted or unreadable format).")
    else:
        print("✅ Image loaded successfully.")
        cv2.imshow("Sample Gesture", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
