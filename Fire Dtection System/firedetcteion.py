import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import local_binary_pattern
from scipy.fft import fft

# Load the HSV reference image or create a default one if file not found
hsv_reference = cv2.imread("hsv-0.png")
if hsv_reference is None:
    # Create a default HSV reference image
    hsv_reference = np.zeros((100, 300, 3), dtype=np.uint8)
    # Add some color information for visualization
    cv2.putText(hsv_reference, "HSV Reference", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
def fire_color_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV ranges for fire colors (blue to yellow to red)
    # Blue flames
    lower_bound1 = np.array([100, 150, 150], dtype=np.uint8)
    upper_bound1 = np.array([120, 255, 255], dtype=np.uint8)
    
    # Yellow flames
    lower_bound2 = np.array([20, 150, 150], dtype=np.uint8)
    upper_bound2 = np.array([35, 255, 255], dtype=np.uint8)
    
    # Red flames
    lower_bound3 = np.array([0, 150, 150], dtype=np.uint8)
    upper_bound3 = np.array([10, 255, 255], dtype=np.uint8)
    
    # Red wrap-around (due to HSV color space)
    lower_bound4 = np.array([170, 150, 150], dtype=np.uint8)
    upper_bound4 = np.array([180, 255, 255], dtype=np.uint8)
    
    # Combine all masks
    mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)  # Blue
    mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)  # Yellow
    mask3 = cv2.inRange(hsv, lower_bound3, upper_bound3)  # Red
    mask4 = cv2.inRange(hsv, lower_bound4, upper_bound4)  # Red wrap-around
    
    # Combine all masks
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)
    mask = cv2.bitwise_or(mask, mask4)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours and filter based on size and shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1500:  # Increased area threshold for better accuracy
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:  # Filter based on aspect ratio to avoid false positives
                # Additional check for flicker or texture to reduce false positives
                roi = mask[y:y+h, x:x+w]
                if np.mean(roi) > 50:  # Ensure intensity is significant
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Fire Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame, mask

def detect_motion(frame, background_subtractor):
    fg_mask = background_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask

from scipy.fft import fft

def check_flicker(intensity_history, sample_rate=30):
    if len(intensity_history) < sample_rate:
        return False
    yf = fft(intensity_history[-sample_rate:])
    xf = np.fft.fftfreq(sample_rate, 1/sample_rate)
    dominant_freq = np.abs(xf[np.argmax(np.abs(yf))])
    return 8 < dominant_freq < 12  # Check for ~10 Hz flicker

from skimage.feature import local_binary_pattern

def analyze_texture(roi_gray):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(roi_gray, n_points, radius, method='uniform')
    return np.histogram(lbp, bins=np.arange(0, n_points + 3), density=True)[0]
def process_frame(frame, background_subtractor, intensity_history):
    # Fire color detection
    _, fire_mask = fire_color_mask(frame)  # Unpack the tuple correctly
    
    # Motion detection
    motion_mask = detect_motion(frame, background_subtractor)
    
    # Combine fire mask and motion mask
    combined_mask = cv2.bitwise_and(fire_mask, motion_mask)
    
    # Get ROI for further analysis
    fire_roi = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    # Flicker analysis
    gray = cv2.cvtColor(fire_roi, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray) if np.any(gray) else 0
    intensity_history.append(intensity)
    is_flickering = check_flicker(intensity_history)
    
    # Draw detection results
    result_frame = frame.copy()
    if is_flickering and np.sum(combined_mask) > 5000:
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1500:  # Increased threshold
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(result_frame, "Fire!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return result_frame, combined_mask

# Main loop
cap = cv2.VideoCapture(0)
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16)
intensity_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize HSV reference image if it exists
    if hsv_reference is not None:
        hsv_ref_resized = cv2.resize(hsv_reference, (300, 100))
    else:
        hsv_ref_resized = np.zeros((100, 300, 3), dtype=np.uint8)
    if not ret:
        break
    processed_frame, mask = fire_color_mask(frame)  # Using the correct function name
    
    # Resize HSV reference image
    hsv_ref_resized = cv2.resize(hsv_reference, (300, 100))
    
    # Display results
    cv2.imshow("Fire Detection", processed_frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("HSV Reference", hsv_ref_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# Initialize
background_subtractor = cv2.createBackgroundSubtractorMOG2()
intensity_history = []

while True:
    ret, frame = cap.read()
    if not ret: break

    # Step 1: Color Masking
    fire_roi = fire_color_mask(frame)

    # Step 2: Motion Detection
    motion_mask = detect_motion(frame, background_subtractor)

    # Step 3: Flicker Analysis
    gray = cv2.cvtColor(fire_roi, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray) if np.any(gray) else 0
    intensity_history.append(intensity)
    is_flickering = check_flicker(intensity_history)

    # Step 4: Texture Analysis
    texture_feature = analyze_texture(gray)

    # Combine Features for Final Decision
    if (np.sum(motion_mask) > 1000 and is_flickering and np.sum(fire_roi) > 5000):
        print("Fire detected!")

    
    def forward(self, x):
        return torch.sigmoid(self.model(x))

def custom_loss(y_pred, y_true, physics_loss_weight=0.1):
    bce_loss = nn.BCELoss()(y_pred, y_true)
    
    # Calculate frequency from predictions using FFT
    y_pred_freq = torch.abs(torch.fft.fftfreq(y_pred.shape[-1], 1/30))
    physics_loss = torch.mean(torch.relu(8 - y_pred_freq) + torch.relu(y_pred_freq - 12))
    
    return bce_loss + physics_loss_weight * physics_loss
# Define the FireDetector model
class FireDetector(nn.Module):
    def __init__(self):
        super(FireDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Define transform for preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Create model instance
model = FireDetector().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = custom_loss

def predict_fire(frame_sequence):
    # Preprocess frames
    frames = [transform(frame) for frame in frame_sequence]
    frames = torch.stack(frames).unsqueeze(0).cuda()
    
    with torch.no_grad():
        prob = model(frames).item()
    
    return prob > 0.5
    return prob > 0.5  # Fire if probability > 50%