#!/bin/bash

# Update package lists
echo "Updating package lists..."
apt update

# Install zip package
echo "Installing zip..."
apt install -y zip

# Install Python packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Setup gdrive
echo "Setting up gdrive..."
tar -xvzf gdrive_linux-x64.tar.gz
chmod +x gdrive
mv gdrive /usr/local/bin/gdrive
gdrive account import gdrive_export-11208120_gs_ncku_edu_tw.tar

# Make directory
mkdir experiments
mkdir data

echo "Environment setup completed!"s
