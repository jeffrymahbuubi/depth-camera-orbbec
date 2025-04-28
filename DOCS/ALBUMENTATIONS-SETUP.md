# Albumentations Library Setup Guide

## Getting Started

- The Albumentations library is known to install `opencv-python-headless` even if the `opencv-python` library is already installed. This happens because the `setup.py` file forces the installation of `opencv-python-headless`. Although the documentation suggests using `pip install -U albumentations --no-binary albumentations` to avoid this issue, further modifications to the `setup.py` files of the `albucore` and `albumentations` libraries are required.

## Setup Instructions

### 1. Modify and Install `albucore`

- Clone the [`albucore`](https://github.com/albumentations-team/albucore) repository and refactor the `setup.py` file as follows:

   ```python
   import re
   from pkg_resources import DistributionNotFound, get_distribution
   from setuptools import setup, find_packages

   INSTALL_REQUIRES = [
       "numpy>=1.24.4",
       "typing-extensions>=4.9.0; python_version<'3.10'",
       "stringzilla>=3.10.4",
       "simsimd>=5.9.2"
   ]

   MIN_OPENCV_VERSION = "4.9.0.80"

   CHOOSE_INSTALL_REQUIRES = [
       (
           (f"opencv-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python-headless>={MIN_OPENCV_VERSION}"),
           None,  # Changed from opencv-python-headless to None
       ),
   ]

   def choose_requirement(mains: tuple[str, ...], secondary: str) -> str:
       # If secondary is None, only try to use existing installations
       if secondary is None:
           for main in mains:
               try:
                   name = re.split(r"[!<>=]", main)[0]
                   get_distribution(name)
                   return main  # Return the found installation requirement
               except DistributionNotFound:
                   pass
           return ""  # Return an empty string if no installation is found

       # Original behavior if secondary is provided
       chosen = secondary
       for main in mains:
           try:
               name = re.split(r"[!<>=]", main)[0]
               get_distribution(name)
               chosen = main
               break
           except DistributionNotFound:
               pass
       return chosen

   def get_install_requirements(install_requires: list[str], choose_install_requires: list[tuple[tuple[str, ...], str]]) -> list[str]:
       for mains, secondary in choose_install_requires:
           requirement = choose_requirement(mains, secondary)
           if requirement:  # Only append if requirement is not empty
               install_requires.append(requirement)
       return install_requires

   setup(
       packages=find_packages(exclude=["tests", "benchmark"], include=['albucore*']),
       install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
   )
   ```

- Install the modified `albucore` library:

   ```bash
   pip install -v -e .
   ```

---

### 2. Modify and Install `albumentations`

- Clone the [`albumentations`](https://github.com/albumentations-team/albumentations) repository and refactor the `setup.py` file as follows:

   ```python
   import re
   from pkg_resources import DistributionNotFound, get_distribution
   from setuptools import setup, find_packages

   INSTALL_REQUIRES = [
       "numpy>=1.24.4",
       "scipy>=1.10.0",
       "PyYAML",
       "typing-extensions>=4.9.0; python_version<'3.10'",
       "pydantic>=2.9.2",
       "albucore==0.0.24",
       "eval-type-backport; python_version<'3.10'",
   ]

   MIN_OPENCV_VERSION = "4.9.0.80"

   # OpenCV packages in order of preference
   OPENCV_PACKAGES = [
       f"opencv-python>={MIN_OPENCV_VERSION}",
       f"opencv-contrib-python>={MIN_OPENCV_VERSION}",
       f"opencv-contrib-python-headless>={MIN_OPENCV_VERSION}",
   ]

   def is_installed(package_name: str) -> bool:
       try:
           get_distribution(package_name)
           return True
       except DistributionNotFound:
           return False

   def choose_opencv_requirement():
       """Check if any OpenCV package is already installed and use that one."""
       # First, try to import cv2 to see if any OpenCV is installed
       try:
           import cv2

           # Try to determine which package provides the installed cv2
           for package in OPENCV_PACKAGES:
               package_name = re.split(r"[!<>=]", package)[0].strip()
               if is_installed(package_name):
                   return package

           # If we can import cv2 but can't determine the package,
           # don't add any OpenCV requirement
           return None

       except ImportError:
           # No OpenCV installed - don't add any requirement (changed from defaulting to headless)
           return None

   # Add OpenCV requirement if needed
   if opencv_req := choose_opencv_requirement():
       INSTALL_REQUIRES.append(opencv_req)

   setup(
       packages=find_packages(exclude=["tests", "tools", "benchmark"], include=['albumentations*']),
       install_requires=INSTALL_REQUIRES,
   )
   ```

- Install the modified `albumentations` library:

   ```bash
   pip install -v -e .
   ```

---

### Summary

By modifying the `setup.py` files of both `albucore` and `albumentations`, you can avoid the forced installation of `opencv-python-headless` and ensure compatibility with your existing OpenCV installation. Follow the steps above to complete the setup process.