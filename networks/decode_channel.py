head_channels = 512
decoder_channels = (256, 128, 64, 16)


in_channels = [head_channels] + list(decoder_channels[:-1])
print(in_channels)