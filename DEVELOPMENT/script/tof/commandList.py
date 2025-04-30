class commandList():
  # setup commands
  COMMAND_SET_ADAPTIVE_INTEGRATION = 0x55                     
  COMMAND_SET_INTEGRATION_TIME_3D = 0x00                      #Command to set the integration time for 3D operation
  COMMAND_SET_PERIOD_FREQUENCY = 0x58
  # acquisition commands
  COMMAND_GET_REGION_DISTANCE_AMPLITUDE = 0x29                #Command to request distance and amplitude data
  COMMAND_GET_REGION_LENGTH_WIDTH = 0x57                      #Command to get the length and width of the region
  # general commands
  COMMAND_SET_POWER = 0x40                                    #Command to enable/disable the power
  COMMAND_GET_CHIP_INFORMATION = 0x48                         #Command to read the chip information
  COMMAND_GET_FIRMWARE_RELEASE = 0x49                         #Command to read the firmware release
  COMMAND_GET_TEMPERATURE = 0x4A

