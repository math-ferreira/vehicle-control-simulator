def list_available_cameras(max_cameras=10):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    return available

# Suppress OpenCV C++ error messages
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import sys
import warnings

def list_available_cameras(max_cameras=10):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    return available

if __name__ == "__main__":
    # Suppress OpenCV error messages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Redirect stderr to null to hide OpenCV errors
        sys.stderr = open(os.devnull, "w")
        cameras = list_available_cameras()
        # Restore stderr
        sys.stderr = sys.__stderr__

    print(f"Available camera indices: {cameras}")

    # Get camera device names using WMI (Windows only)
    try:
        import wmi
        c = wmi.WMI()
        print("\nCamera device names detected:")
        for device in c.Win32_PnPEntity():
            if device.Name and ("Camera" in device.Name or "Webcam" in device.Name or "Imaging" in device.Name):
                print(f"- {device.Name}")
    except ImportError:
        print("wmi module not installed. Cannot list camera device names.")
    except Exception as e:
        print(f"Error retrieving camera names: {e}")
