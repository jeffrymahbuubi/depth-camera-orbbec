# ORBBEC Gemini 2XL Depth Camera Setup

- ORBBEC Python SDK Official Documentation: [https://orbbec.github.io/pyorbbecsdk/index.html](https://orbbec.github.io/pyorbbecsdk/index.html)
- Useful Links: 
    - [ORBBEC Depth Camera (I): Environment Configuration](https://blog.csdn.net/WYKB_Mr_Q/article/details/137040226)
    - [ORBBEC Depth Camera (II) PyQt5 Development](https://blog.csdn.net/WYKB_Mr_Q/article/details/137084563?spm=1001.2014.3001.5501)

## Windows Setup Guide

### 1. Create a New Environment (Preferably NOT using Anaconda)
   - Clone the SDK repository:
     ```bash
     git clone https://github.com/orbbec/pyorbbecsdk.git
     cd pyorbbecsdk
     ```
   - Create a new Python virtual environment:
     ```bash
     python -m venv ./venv
     ```
   - Activate the virtual environment:
     ```powershell
     .\venv\Scripts\activate
     ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

### 2. Create a Copy of the `pyorbbecsdk` Repository
   - To preserve the original version of the SDK, copy the `pyorbbecsdk` repository and rename it:
     ```bash
     cp -r pyorbbecsdk pyorbbec-sdk-experiments
     ```
   - Navigate to the new folder:
     ```bash
     cd pyorbbec-sdk-experiments
     ```
   - Create a new Python virtual environment within this directory:
     ```bash
     python -m venv ./venv
     ```

### 3. Build the Python Binding Using CMake
   - Download and install the following tools:
     - [Visual Studio 2022](https://visualstudio.microsoft.com/)
     - [CMake GUI](https://cmake.org/download/)
   - Open the CMake GUI and configure the build:
     1. **Set paths:**
        - `Where is the source code` ➡️ Point to the `pyorbbec-sdk-experiments` directory.
        - `Where to build the binaries` ➡️ Create a new folder named `build` inside `pyorbbec-sdk-experiments` and point to this `build` folder.
     2. **Add CMake entries:**
        - Click **Add Entry** and provide the following entries:
          - `pybind11_DIR` ➡️ Point to `venv/share/cmake/pybind11`.
          - `BUILD_TESTING` ➡️ Set type to `BOOLEAN` and value to `OFF` (if applicable).
     3. **Configure the project:**
        - Click **Configure** and choose:
          - **Generator:** Select your Visual Studio version.
          - **Platform:** Type `x64`.
        - Click **Finish** when prompted.
     4. **Generate the build files:**
        - After seeing `Configuring done`, click **Generate** and wait for `Generating done`.
        - Click **Open Project**, which will launch Visual Studio with the solution file (`pyorbbecsdk.sln`) located in the `build` folder.

### 4. Visual Studio Configuration
   - In Visual Studio:
     1. Set the build configuration to **Release** and **x64**.
     2. In the Solution Explorer:
        - Right-click on the `pyorbbecsdk` project and select **Rebuild**.
        - After rebuilding, right-click on the `INSTALL` target and select **Build**.
   - Navigate to the installation folder:
     ```bash
     cd pyorbbec-sdk-experiments/install/lib
     ```
   - Copy all files from this folder into the `examples` directory:
     ```bash
     cp * ../examples
     ```

### 5. Run Examples
   - Navigate to the `examples` directory:
     ```bash
     cd pyorbbec-sdk-experiments/examples
     ```
   - Run any example script provided in the folder:
     ```bash
     python <example_script.py>
     ```

Now you're ready to use the ORBBEC Gemini 2XL Depth Camera with the Python SDK!

