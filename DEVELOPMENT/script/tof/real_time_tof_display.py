#!/usr/bin/env python3
"""
Real-time visualization of TOF sensor data (distance and amplitude)
using OpenCV for the P8864 3D TOF module with frame rate control.
"""

import time
import argparse
import logging
import numpy as np
import cv2
from tofCam611 import TOFcam611

# Set up logging with less verbose output for real-time display
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tof_display')

# Constants for visualization
WINDOW_NAME = "TOF Sensor Data"
MAX_EXPECTED_DISTANCE = 3000  # 3 meters in mm
MIN_EXPECTED_DISTANCE = 100   # 10 cm in mm
ERROR_MIN_AMPLITUDE = 16001000  # Error code for pixels with amplitude below threshold

# Color maps for visualization
DISTANCE_COLORMAP = cv2.COLORMAP_JET
AMPLITUDE_COLORMAP = cv2.COLORMAP_INFERNO

# Display modes
DISPLAY_MODE_DISTANCE = 0
DISPLAY_MODE_BOTH = 1
DISPLAY_MODE_NAMES = ["Distance Only", "Distance & Amplitude"]

def normalize_image(image, min_val, max_val):
    """Normalize image to 0-255 range for display"""
    # Clip values to the expected range
    clipped = np.clip(image, min_val, max_val)
    
    # Normalize to 0-255
    normalized = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def create_display_image(data, colormap, title, scale=30, add_text_banner=True):
    """Create a display image with color mapping and text"""
    # Get data dimensions
    height, width = data.shape
    
    # Resize for better visibility (nearest neighbor to maintain pixel boundaries)
    display = cv2.resize(data, (width*scale, height*scale), interpolation=cv2.INTER_NEAREST)
    
    # Apply colormap
    display_color = cv2.applyColorMap(display, colormap)
    
    # Add text banner if requested
    if add_text_banner:
        # Create a black banner at the top for text
        banner_height = 30
        text_banner = np.zeros((banner_height, display_color.shape[1], 3), dtype=np.uint8)
        
        # Add title to the banner
        cv2.putText(text_banner, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Combine banner with display image
        display_color = np.vstack((text_banner, display_color))
    
    return display_color

def add_pixel_values(image, data, scale=30, banner_height=30):
    """Add pixel values as text overlays to the image"""
    height, width = data.shape
    for y in range(height):
        for x in range(width):
            value = data[y, x]
            if value == ERROR_MIN_AMPLITUDE:
                text = "N/A"
            else:
                text = f"{int(value)}"
            
            # Calculate position for text (center of each scaled pixel)
            pos_x = int((x + 0.5) * scale)
            pos_y = int((y + 0.5) * scale) + banner_height  # Adjust for banner
            
            # Choose text color based on background
            # Small shadow to make text more readable on any background
            cv2.putText(image, text, (pos_x-1, pos_y-1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, text, (pos_x, pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image

def create_visualization(distance, amplitude, show_values=True, scale=30, display_mode=DISPLAY_MODE_BOTH):
    """Create visualization based on the selected display mode"""
    # Mask for invalid distance values
    invalid_mask = (distance == ERROR_MIN_AMPLITUDE)
    
    # Create normalized versions for display
    distance_display = distance.copy()
    distance_display[invalid_mask] = MAX_EXPECTED_DISTANCE
    
    distance_norm = normalize_image(distance_display, MIN_EXPECTED_DISTANCE, MAX_EXPECTED_DISTANCE)
    
    # Get min/max for distance display
    if np.any(~invalid_mask):
        min_dist = np.min(distance[~invalid_mask])
        max_dist = np.max(distance[~invalid_mask])
        distance_title = f"Distance (mm) - Min: {min_dist:.1f}, Max: {max_dist:.1f}"
    else:
        distance_title = "Distance (mm) - No valid data"
    
    # Create distance display
    distance_color = create_display_image(distance_norm, DISTANCE_COLORMAP, distance_title, scale)
    
    # Add pixel values if requested
    if show_values:
        distance_color = add_pixel_values(distance_color, distance, scale)
    
    # If distance-only mode, return just the distance visualization
    if display_mode == DISPLAY_MODE_DISTANCE:
        return distance_color
    
    # If both, create amplitude visualization too
    amplitude_norm = normalize_image(amplitude, 0, np.percentile(amplitude, 95))  # Use percentile to avoid extreme outliers
    amplitude_title = f"Amplitude - Min: {np.min(amplitude)}, Max: {np.max(amplitude)}"
    amplitude_color = create_display_image(amplitude_norm, AMPLITUDE_COLORMAP, amplitude_title, scale)
    
    # Add pixel values if requested
    if show_values:
        amplitude_color = add_pixel_values(amplitude_color, amplitude, scale)
    
    # Combine images side by side
    # Make sure both images have the same height
    if distance_color.shape[0] != amplitude_color.shape[0]:
        max_height = max(distance_color.shape[0], amplitude_color.shape[0])
        if distance_color.shape[0] < max_height:
            pad = np.zeros((max_height - distance_color.shape[0], distance_color.shape[1], 3), dtype=np.uint8)
            distance_color = np.vstack((distance_color, pad))
        if amplitude_color.shape[0] < max_height:
            pad = np.zeros((max_height - amplitude_color.shape[0], amplitude_color.shape[1], 3), dtype=np.uint8)
            amplitude_color = np.vstack((amplitude_color, pad))
    
    combined = np.hstack((distance_color, amplitude_color))
    return combined

def add_info_panel(image, info_text):
    """Add an information panel at the bottom of the image"""
    # Create a black banner at the bottom for information
    banner_height = 60
    info_banner = np.zeros((banner_height, image.shape[1], 3), dtype=np.uint8)
    
    # Add text lines to the info banner
    y_pos = 20
    line_height = 20
    for line in info_text:
        cv2.putText(info_banner, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_pos += line_height
    
    # Combine image with info banner
    return np.vstack((image, info_banner))

def run_visualization(tof_cam, fps_target=10, show_values=True, fps_mode='medium', display_mode=DISPLAY_MODE_BOTH):
    """Run the main visualization loop"""
    # Initialize FPS calculation variables
    frame_count = 0
    start_time = time.time()
    actual_fps = 0
    
    # Initialize sensor with reasonable settings
    tof_cam.initialize()
    tof_cam.settings.set_minimal_amplitude(100)  # Set minimum amplitude threshold
    
    # Configure FPS mode using the period frequency setting
    if hasattr(tof_cam.settings, 'set_fps_mode'):
        tof_cam.settings.set_fps_mode(fps_mode)
    else:
        # Fallback if method is not implemented yet
        logger.warning("set_fps_mode not implemented in TOFcam611_Settings class")
        # Direct call to set_period_frequency with appropriate value
        period = {"high": 9, "medium": 5, "accurate": 2}.get(fps_mode, 1)
        tof_cam.settings.set_period_frequency(period)
    
    # Information for the user
    logger.info(f"Starting visualization with {fps_mode} FPS mode")
    logger.info(f"Display mode: {DISPLAY_MODE_NAMES[display_mode]}")
    logger.info("Controls: press 'q' to quit, 's' to save frame, 'v' to toggle values")
    logger.info("          press 'f' to cycle through FPS modes")
    logger.info("          press 'd' to toggle display mode")
    
    # FPS mode cycle list
    fps_modes = ['low', 'accurate', 'medium', 'high']
    current_mode_idx = fps_modes.index(fps_mode) if fps_mode in fps_modes else 1
    
    try:
        while True:
            loop_start = time.time()
            
            # Get distance and amplitude images
            try:
                distance = tof_cam.get_distance_image()
                amplitude = tof_cam.get_amplitude_image()
            except Exception as e:
                logger.error(f"Error getting data from sensor: {e}")
                # Brief pause before retrying
                time.sleep(0.5)
                continue
            
            # Create visualization based on selected display mode
            display = create_visualization(distance, amplitude, show_values, 
                                          display_mode=display_mode)
            
            # Add information panel
            period_info = getattr(tof_cam.settings, '_period_frequency', '?')
            info_text = [
                f"Mode: {fps_mode} (period: {period_info}) | FPS: {actual_fps:.1f}",
                f"Controls: [Q]uit | [S]ave | [V]alues | [F]PS mode"
            ]
            display = add_info_panel(display, info_text)
            
            # Show the image
            cv2.imshow(WINDOW_NAME, display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('s'):  # 's' to save current frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"tof_capture_{timestamp}.png", display)
                logger.info(f"Frame saved as tof_capture_{timestamp}.png")
            elif key == ord('v'):  # 'v' to toggle value display
                show_values = not show_values
                logger.info(f"Value display: {'on' if show_values else 'off'}")
            elif key == ord('f'):  # 'f' to cycle through FPS modes
                current_mode_idx = (current_mode_idx + 1) % len(fps_modes)
                fps_mode = fps_modes[current_mode_idx]
                
                # Update the camera's FPS mode
                if hasattr(tof_cam.settings, 'set_fps_mode'):
                    tof_cam.settings.set_fps_mode(fps_mode)
                else:
                    # Fallback
                    period = {"high": 9, "medium": 5, "accurate": 2, "low": 1}.get(fps_mode, 1)
                    tof_cam.settings.set_period_frequency(period)
                
                logger.info(f"Switched to {fps_mode} FPS mode")
            elif key == ord('d'):  # 'd' to toggle display mode
                display_mode = (display_mode + 1) % len(DISPLAY_MODE_NAMES)
                logger.info(f"Switched to {DISPLAY_MODE_NAMES[display_mode]} display mode")
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:  # Update FPS every second
                actual_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Frame rate control for display (this doesn't affect sensor frame rate)
            display_elapsed = time.time() - loop_start
            if display_elapsed < 1.0/fps_target:
                time.sleep(1.0/fps_target - display_elapsed)
                
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user")
    finally:
        # Clean up
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Real-time TOF sensor visualization")
    parser.add_argument('--port', type=str, help='Serial port to connect to (optional)', default=None)
    parser.add_argument('--fps', type=int, help='Target display FPS', default=10)
    parser.add_argument('--scale', type=int, help='Display scale factor', default=30)
    parser.add_argument('--fps-mode', type=str, choices=['low', 'accurate', 'medium', 'high'], 
                        help='Camera FPS mode', default='medium')
    parser.add_argument('--no-values', action='store_true', help='Hide pixel values in visualization')
    parser.add_argument('--display', type=str, choices=['distance', 'both'], 
                        help='Display mode', default='both')
    args = parser.parse_args()
    
    # Convert display mode string to constant
    display_mode = DISPLAY_MODE_DISTANCE if args.display == 'distance' else DISPLAY_MODE_BOTH
    
    logger.info("Starting TOF visualization")
    
    try:
        # Initialize the TOF camera
        logger.info(f"Connecting to TOF camera{' on '+args.port if args.port else ''}")
        tof_cam = TOFcam611(port=args.port)
        logger.info("Connection established")
        
        # Run visualization
        run_visualization(
            tof_cam, 
            fps_target=args.fps, 
            show_values=not args.no_values,
            fps_mode=args.fps_mode,
            display_mode=display_mode
        )
        
        logger.info("Visualization completed")
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())