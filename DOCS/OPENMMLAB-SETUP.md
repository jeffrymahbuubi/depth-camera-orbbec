# OpenMMLab Setup Guide

## Getting Started

- GitHub Repository: [OpenMMLab](https://github.com/open-mmlab)

## Environment Setup

### MMPose

Follow the steps below to set up the environment for MMPose:

1. **Install `openmim` and `mmengine`**:

   ```bash
   pip install -U openmim
   mim install mmengine
   ```

2. **Install Visual Studio**:

   - Download Visual Studio 2019/2022 Community Edition.
   - Ensure the `x64 Native Tools Command Prompt for VS 2022` compiler is installed.
   - Refer to the [Visual Studio Release Notes](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#installing-an-earlier-release) for installation details.

3. **Clone the Repositories**:

   - Clone the following repositories:
     - [mmcv](https://github.com/open-mmlab/mmcv)
     - [mmpose](https://github.com/open-mmlab/mmpose)

4. **Install `mmcv` from Source**:

   - Open the `x64 Native Tools Command Prompt for VS 2022`.
   - Navigate to the `mmcv` folder and follow the [installation guide](https://mmcv.readthedocs.io/en/latest/get_started/build.html).

   > **Note**: Ensure you follow all the steps outlined in the documentation.

5. **Install `mmdet`**:

   ```bash
   mim install "mmdet>=3.0.0"
   ```

   - After installation, navigate to `venv\Lib\site-packages\mmdet\__init__.py` and update the `mmcv_maximum_version` to `2.3.0` to avoid compatibility issues.

6. **Install `mmpose`**:

   - Clone the `chumpy` library from [here](https://github.com/mattloper/chumpy).
   - Refactor the `setup.py` file in the `chumpy` directory as follows:

     ```python
     """
     Author(s): Matthew Loper

     See LICENCE.txt for licensing and contact information.
     """

     from setuptools import setup
     from runpy import run_path

     # Simple approach to read requirements instead of using parse_requirements
     with open('requirements.txt') as f:
         install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

     def get_version():
         namespace = run_path('chumpy/version.py')
         return namespace['version']

     setup(
         name='chumpy',
         version=get_version(),
         packages=['chumpy'],
         author='Matthew Loper',
         author_email='matt.loper@gmail.com',
         url='https://github.com/mattloper/chumpy',
         description='chumpy',
         license='MIT',
         install_requires=install_requires,
         classifiers=[
             'Development Status :: 4 - Beta',
             'Intended Audience :: Science/Research',
             'Topic :: Scientific/Engineering :: Mathematics',
             'License :: OSI Approved :: MIT License',
             'Programming Language :: Python :: 2',
             'Programming Language :: Python :: 2.7',
             'Programming Language :: Python :: 3',
             'Operating System :: MacOS :: MacOS X',
             'Operating System :: POSIX :: Linux'
         ],
     )
     ```

   - Open the `x64 Native Tools Command Prompt for VS 2022`.
   - Navigate to the `mmpose` directory and install it using the following commands:

     ```bash
     pip install -r requirements.txt
     pip install -v -e .
     ```

   > **Note**: Run `pip check` to ensure there are no errors related to any OpenMMLab libraries.

7. **Fix Known Issues in the Demo**:

   - In the demo outlined in the [MMPose documentation](https://mmpose.readthedocs.io/en/latest/guide_to_framework.html), some functions may cause errors. Follow the solution below to fix them:

     - Update the `mmpose\visualization\local_visualizer_3d.py` file:

       ```python
       # Change from:
       pred_img_data = pred_img_data.reshape(
           int(height),
           int(width) * num_instances, 3)

       # To:
       pred_img_data = pred_img_data.reshape(
           int(height),
           int(width) * num_instances, 4)  # Use 4 for RGBA
       # Then convert to RGB by dropping the alpha channel
       pred_img_data = pred_img_data[:, :, :3]
       ```