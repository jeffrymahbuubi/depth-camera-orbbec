import time
import struct
import logging
import numpy as np
from typing import Optional
from communicationType import communicationType as ComType
from commandList import commandList as CommandList
# Fix the import - need to import the class directly from the module
from serialInterface import SerialInterface as SerialInterfaceClass
from crc import Crc, CrcMode
from transformations_3d import depth_to_3d
import atexit

# Constants based on P8864 TOF module datasheet
MAX_INTEGRATION_TIME = 2**16-1  # Maximum integration time value
ERROR_MIN_AMPLITUDE = 16001000  # Error code for pixels with amplitude below threshold
CONVERT_TO_MM = 10              # Convert from 0.1mm to mm
DEFAULT_MAX_DEPTH = 16000       # Default maximum depth value

log = logging.getLogger('TOFcam611')

class Dev_Infos_Controller:
    def __init__(self) -> None:
        pass

    def get_chip_temperature(self) -> float:
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_chip_temperature' jet")

    def get_chip_infos(self) -> tuple[int, int]:
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_chip_id' jet")

    def get_fw_version(self) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_fw_version' jet")

class TOF_Settings_Controller:
    def __init__(self) -> None:
        pass
    
    def set_modulation(self, frequency_mhz: float, channel=0):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'set_modulation_frequency' jet")
    
    def get_modulation_frequencies(self) -> list[float]:
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_modulation_frequencies' jet")
    
    def get_modulation_channels(self) -> list[int]:
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_modulation_channels' jet")
    
    def get_roi(self):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_roi' jet")

    def set_roi(self, roi: tuple[int, int, int, int]):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'set_roi' jet")

    def set_minimal_amplitude(self, amplitude: int):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'set_minimal_amplitude' jet")
    
    def set_integration_time(self, int_time_us: int):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'set_integration_time' jet")

    def set_integration_time_grayscale(self, int_time_us: int):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'set_integration_time_grayscale' jet")

class InterfaceWrapper:
    """Wrapper for serial communication with the TOF module"""
    def __init__(self, port: Optional[str]=None) -> None:
        # Use the class directly, not the module
        self.com = SerialInterfaceClass(port, baudrate=115200, timeout=2.0)  # Increased timeout
        self.crc = Crc(mode=CrcMode.CRC32_STM32, revout=False)
        self.__answer_table = {
            ComType.DATA_TEMPERATURE: 10,
            ComType.DATA_FIRMWARE_RELEASE: 12,
            ComType.DATA_REGION_LENGTH_WIDTH: 10,
        }

    def __get_answer_len(self, ret_type: int) -> int:
        """Get expected answer length for a given response type"""
        try:
            return self.__answer_table[ret_type]
        except KeyError:
            raise ValueError(f"Return type '0x{ret_type:02X}' not supported")

    def tofWrite(self, values) -> None:
        """Format and send a command to the TOF module"""
        if type(values)!=list:
            values = [values]
        values += [0] * (9 - len(values))  # Fill up to size 9 with zeros
        a = [0xF5] * 14
        a[1:10] = values

        # Calculate CRC32 checksum
        crc = np.array(self.crc.calculate(bytearray(a[:10])))
        
        a[10] = crc & 0xFF
        a[11] = (crc >> 8) & 0xFF
        a[12] = (crc >> 16) & 0xFF
        a[13] = (crc >> 24) & 0xFF

        # Log the command being sent
        cmd_hex = ' '.join([f'0x{x:02X}' for x in a])
        log.debug(f"Sending command: {cmd_hex}")
        
        self.com.write(a)
        time.sleep(0.1)  # Add a small delay after sending command

    def getAcknowledge(self):
        """Read and verify acknowledge response"""
        LEN_BYTES = 8
        tmp = self.com.read(LEN_BYTES)
        
        if len(tmp) != LEN_BYTES:
            log.error(f"Acknowledge: Wrong number of bytes received! Expected {LEN_BYTES}, got {len(tmp)}")
            # If we received any bytes, try to decode them for debugging
            if len(tmp) > 0:
                log.error(f"Received bytes: {' '.join([f'0x{b:02X}' for b in tmp])}")
            raise Exception(f"Wrong number of bytes received! Expected {LEN_BYTES}, got {len(tmp)}")
            
        if not self.crc.verify(tmp[:-4], tmp[-4:]):
            log.error("Acknowledge: CRC not valid!")
            raise Exception("CRC not valid!")
            
        if tmp[1] != 0x00:  # DATA_ACK value
            log.error(f"Acknowledge: Got wrong type: 0x{tmp[1]:02X}")
            raise Exception(f"Got wrong type: 0x{tmp[1]:02X}")
            
        return True
    
    def getAnswer(self, typeId, length):
        """Read and verify a response of fixed length"""
        # First, read header bytes to see if there's any response
        header = self.com.read(4)
        if len(header) == 0:
            log.error(f"No response received when expecting type 0x{typeId:02X}")
            # Try sending the command again
            raise Exception(f"No response received from device")
            
        if len(header) < 4:
            log.error(f"Incomplete header received: {header}")
            raise Exception(f"Incomplete header received: {header}")
            
        # Now read the rest
        rest = self.com.read(length - 4)
        tmp = header + rest
        
        log.debug(f"Response received: {' '.join([f'0x{b:02X}' for b in tmp])}")
        
        if len(tmp) != length:
            log.error(f"Wrong number of bytes received! Expected {length}, got {len(tmp)}")
            raise Exception(f"Wrong number of bytes received! Expected {length}, got {len(tmp)}")
            
        if not self.crc.verify(tmp[:-4], tmp[-4:]):
            log.error("CRC not valid!")
            raise Exception("CRC not valid!")
            
        if tmp[1] == ComType.DATA_NACK:
            log.error("Received NACK")
            raise Exception("Received NACK")
            
        if typeId != tmp[1]:
            log.error(f"Wrong Type! Expected 0x{typeId:02X}, got 0x{tmp[1]:02X}")
            raise Exception(f"Wrong Type! Expected 0x{typeId:02X}, got 0x{tmp[1]:02X}")
            
        data_length = struct.unpack('<'+'H', tmp[2:4])[0]
        return tmp[4:4+data_length]
       
    def getData(self, typeId):
        """Read data response with variable length"""
        tmp = self.com.read(4)
        if len(tmp) == 0:
            log.error(f"No response received when expecting type 0x{typeId:02X}")
            raise Exception(f"No response received from device")
            
        total = bytes(tmp)
        length = struct.unpack('<'+'H', tmp[2:4])[0]
        tmp = self.com.read(length+4)
        total += bytes(tmp)
        
        if len(tmp) < length+4:
            log.error(f"Incomplete data received. Expected {length+4} more bytes, got {len(tmp)}")
            raise Exception(f"Incomplete data received")
            
        self.crc.verify(bytearray(total[:-4]), bytearray(total[-4:]))
        if typeId != total[1]:
            log.error(f"Wrong Type! Expected 0x{typeId:02X}, got 0x{total[1]:02X}")
            raise Exception(f"Wrong Type! Expected 0x{typeId:02X}, got 0x{total[1]:02X}")

        return [tmp[:-4], length]

    def transmit(self, cmd_id: int, arg=[]):
        """Send command with arguments and wait for acknowledgement"""
        arg_copy = arg.copy()  # Create copy to avoid modifying the original
        arg_copy.insert(0, cmd_id)
        self.tofWrite(arg_copy)
        self.getAcknowledge()

    def transceive(self, cmd_id: int, response_id: int, arg=[]):
        """Send command and receive response"""
        arg_copy = arg.copy()  # Create copy to avoid modifying the original
        arg_copy.insert(0, cmd_id)
        self.tofWrite(arg_copy)
        
        # Add delay before reading response
        time.sleep(0.2)
        
        length = self.__get_answer_len(response_id)
        return self.getAnswer(response_id, length)

class TOFcam611_Settings(TOF_Settings_Controller):
    """Settings controller for P8864 TOF module"""
    def __init__(self, interface: InterfaceWrapper) -> None:
        self.interface = interface
        self._min_amplitude = 100  # Default minimum amplitude threshold
        self.maxDepth = DEFAULT_MAX_DEPTH
        self.resolution = self._get_resolution()

    def _get_resolution(self):
        """Get sensor resolution from the module"""
        response = self.interface.transceive(CommandList.COMMAND_GET_REGION_LENGTH_WIDTH, 
                                             ComType.DATA_REGION_LENGTH_WIDTH)
        width, height = struct.unpack('<BB', response)
        return (width, height)  # 8x8 for the P8864 module

    def get_roi(self):
        """Get the region of interest (whole sensor area)"""
        return (0, 0, self.resolution[0], self.resolution[1])

    def set_minimal_amplitude(self, amplitude: int):
        """Set minimum amplitude threshold for distance measurement"""
        log.info(f"Set minimal amplitude to {amplitude}")
        self._min_amplitude = amplitude

    def set_integration_time(self, int_time_us: int):
        """Set integration time for distance measurement in microseconds"""
        if int_time_us <= 0 or int_time_us > MAX_INTEGRATION_TIME:
            raise ValueError(f"Integration time must be between 1 and {MAX_INTEGRATION_TIME} us")
        log.info(f"Set integration time to {int_time_us} us")
        self.interface.transmit(CommandList.COMMAND_SET_INTEGRATION_TIME_3D, 
                              [0, int_time_us & 0xFF, (int_time_us >> 8) & 0xFF])

    def set_adaptive_integration(self, enable: bool = True):
        """Enable or disable adaptive integration time feature"""
        log.info(f"{'Enable' if enable else 'Disable'} adaptive integration time")
        self.interface.transmit(CommandList.COMMAND_SET_ADAPTIVE_INTEGRATION, 
                              [0 if enable else 1])

    def set_period_frequency(self, period: int = 1):
        """Set period between adaptive integration time updates"""
        if period < 1 or period > 9:
            raise ValueError(f"Period must be between 1 and 9")
        log.info(f"Set period frequency to {period}")
        self.interface.transmit(CommandList.COMMAND_SET_PERIOD_FREQUENCY, [period])

class TOFcam611_Device(Dev_Infos_Controller):
    """Device information controller for P8864 TOF module"""
    def __init__(self, interface: InterfaceWrapper) -> None:
        self.interface = interface
    
    def get_chip_temperature(self) -> float:
        """Get the chip temperature in °C"""
        response = self.interface.transceive(CommandList.COMMAND_GET_TEMPERATURE, 
                                           ComType.DATA_TEMPERATURE)
        temperature = struct.unpack('<'+'h', response)[0]
        return float(temperature) / 100.0  # Convert to degrees Celsius

    def get_fw_version(self) -> str:
        """Get firmware version as 'major.minor'"""
        response = self.interface.transceive(CommandList.COMMAND_GET_FIRMWARE_RELEASE, 
                                           ComType.DATA_FIRMWARE_RELEASE)
        fw_release = struct.unpack('<'+'I', response)[0]
        major = fw_release >> 16
        minor = fw_release & 0xFFFF
        return f'{major}.{minor}'
    
    def set_power(self, enable=True):
        """Set power state (on/off)"""
        log.info(f"Set power {'on' if enable else 'off'}")
        self.interface.tofWrite([CommandList.COMMAND_SET_POWER, int(enable)])
        time.sleep(1 if enable else 0.03)  # Wait for power-up/down as per datasheet
        self.interface.getAcknowledge()

class TOFcam:
    def __init__(self, settings_ctrl: TOF_Settings_Controller, info_ctrl: Dev_Infos_Controller) -> None:
        self.settings = settings_ctrl
        self.device = info_ctrl
        atexit.register(self.__del__)

    def initialize(self):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'initialize' jet")

    def get_distance_image(self):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_distance_image' jet")

    def get_amplitude_image(self):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_amplitude_image' jet")

    def get_grayscale_image(self):
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented 'get_grayscale_image' jet")

class TOFcam611(TOFcam):
    """Main class for P8864 3D TOF module"""
    
    def __init__(self, port: Optional[str]=None) -> None:
        self.interface = InterfaceWrapper(port)
        self.device = TOFcam611_Device(self.interface)
        self.settings = TOFcam611_Settings(self.interface)
        super().__init__(self.settings, self.device)

    def __del__(self):
        """Clean up resources on deletion"""
        if hasattr(self, 'interface') and hasattr(self.interface, 'com') and self.interface.com is not None:
            self.interface.com.close()

    def initialize(self):
        """Initialize the TOF module with default settings"""
        log.info("Initialize P8864 TOF module")
        self.device.set_power(True)
        self.settings.set_integration_time(50)
        self.settings.set_adaptive_integration(True)

    def get_distance_and_amplitude_image(self):
        """Get distance and amplitude data from the TOF module"""
        self.interface.tofWrite([CommandList.COMMAND_GET_REGION_DISTANCE_AMPLITUDE])
        data, length = self.interface.getData(ComType.DATA_DISTANCE_AMPLITUDE)
        
        # Calculate number of pixels from data length
        pixel_count = length // 8  # Each pixel has 4 bytes for distance and 4 bytes for amplitude
        
        # Reshape for the P8864 module (8x8 array)
        width, height = self.settings.resolution
        
        # Extract distance and amplitude data
        dist_data = data[:length//2]
        ampl_data = data[length//2:]
        
        # Unpack and reshape the data
        dist_raw = np.array(struct.unpack('<'+'I'*(pixel_count), dist_data))
        ampl_raw = np.array(struct.unpack('<'+'I'*(pixel_count), ampl_data))
        
        distance = np.reshape(dist_raw, (height, width))
        amplitude = np.reshape(ampl_raw, (height, width))
        
        # Convert from 0.1mm to mm
        return distance / CONVERT_TO_MM, amplitude

    def get_distance_image(self):
        """Get distance image with amplitude thresholding"""
        dist, ampl = self.get_distance_and_amplitude_image()
        
        # Mark pixels with amplitude below threshold as invalid
        dist[ampl < self.settings._min_amplitude] = ERROR_MIN_AMPLITUDE
        
        return dist

    def get_amplitude_image(self):
        """Get amplitude image"""
        _, amplitude = self.get_distance_and_amplitude_image()
        return amplitude

    def get_point_cloud(self):
        """Generate 3D point cloud from distance data"""
        # Get distance image and apply thresholding
        depth = self.get_distance_image()
        depth = depth.astype(np.float32)
        depth[depth >= self.settings.maxDepth] = np.nan
        
        # Calculate point cloud (convert mm to meters with 1E-3 factor)
        points = 1E-3 * depth_to_3d(np.fliplr(depth), 
                                    resolution=self.settings.resolution, 
                                    focalLengh=40)  # Focal length in px
        
        # Reshape to Nx3 point array
        points = np.transpose(points, (1, 2, 0))
        points = points.reshape(-1, 3)
        
        return points