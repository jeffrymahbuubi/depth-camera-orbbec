#!/usr/bin/env python3
"""
Testing script for P8864 3D TOF module using TOFcam611 implementation.
This script demonstrates basic connection, information retrieval, and operations.
"""

import time
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tofCam611 import TOFcam611

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')

def print_device_info(tof_cam):
    """Print basic device information"""
    logger.info("========== DEVICE INFORMATION ==========")
    
    # Get firmware version
    fw_version = tof_cam.device.get_fw_version()
    logger.info(f"Firmware Version: {fw_version}")
    
    # Get chip temperature
    temperature = tof_cam.device.get_chip_temperature()
    logger.info(f"Chip Temperature: {temperature:.2f}°C")
    
    # Get resolution
    resolution = tof_cam.settings.resolution
    logger.info(f"Sensor Resolution: {resolution[0]}x{resolution[1]}")
    
    logger.info("=======================================")

def test_device_control(tof_cam):
    """Test basic device control functions"""
    logger.info("Testing device control...")
    
    # Power off
    logger.info("Powering off...")
    tof_cam.device.set_power(False)
    time.sleep(1)
    
    # Power on
    logger.info("Powering on...")
    tof_cam.device.set_power(True)
    time.sleep(1)
    
    # Set integration time
    logger.info("Setting integration time to 100µs...")
    tof_cam.settings.set_integration_time(100)
    time.sleep(0.5)
    
    # Set adaptive integration
    logger.info("Enabling adaptive integration...")
    tof_cam.settings.set_adaptive_integration(True)
    time.sleep(0.5)
    
    # Set period frequency
    logger.info("Setting period frequency to 2...")
    tof_cam.settings.set_period_frequency(2)
    
    logger.info("Device control tests completed.")

def display_images(tof_cam):
    """Capture and display distance and amplitude images"""
    logger.info("Capturing images...")
    
    # Set minimum amplitude threshold
    tof_cam.settings.set_minimal_amplitude(100)
    
    # Get distance and amplitude images
    distance = tof_cam.get_distance_image()
    amplitude = tof_cam.get_amplitude_image()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot distance image
    dist_img = ax1.imshow(distance, cmap='viridis')
    ax1.set_title('Distance (mm)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(dist_img, ax=ax1, label='Distance (mm)')
    
    # Plot amplitude image
    amp_img = ax2.imshow(amplitude, cmap='inferno')
    ax2.set_title('Amplitude')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(amp_img, ax=ax2, label='Amplitude')
    
    plt.tight_layout()
    plt.savefig('tof_images.png')
    logger.info("Images saved to 'tof_images.png'")
    
    # Try to display if running in interactive environment
    try:
        plt.show()
    except:
        logger.info("Non-interactive environment detected. Plot saved to file only.")

def main():
    parser = argparse.ArgumentParser(description="Test script for P8864 3D TOF module")
    parser.add_argument('--port', type=str, help='Serial port to connect to (optional)', default=None)
    parser.add_argument('--info-only', action='store_true', help='Only display device information')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting operations')
    args = parser.parse_args()
    
    logger.info("Starting P8864 TOF module test")
    
    try:
        # Initialize the TOF camera
        logger.info(f"Connecting to TOF camera{' on '+args.port if args.port else ''}")
        tof_cam = TOFcam611(port=args.port)
        logger.info("Connection established")
        
        # Initialize device
        tof_cam.initialize()
        logger.info("Device initialized")
        
        # Print device information
        print_device_info(tof_cam)
        
        if not args.info_only:
            # Test device control
            test_device_control(tof_cam)
            
            if not args.no_plots:
                # Display images
                display_images(tof_cam)
        
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())