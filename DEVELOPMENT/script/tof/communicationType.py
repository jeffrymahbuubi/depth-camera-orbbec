class communicationType():
  DATA_NACK                    = 0x01     # Not acknowledge from sensor to host
  DATA_ERROR                   = 0xFF     # Error number
  DATA_FIRMWARE_RELEASE        = 0xFE     # Firmware release number
  DATA_TEMPERATURE             = 0xFC     # Temperature data
  DATA_DISTANCE_AMPLITUDE      = 0x0B     # Distance and amplitude data
  DATA_REGION_LENGTH_WIDTH     = 0x0C     # Region length and width data


