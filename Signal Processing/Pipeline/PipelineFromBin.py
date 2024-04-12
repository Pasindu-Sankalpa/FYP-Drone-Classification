import os
import cv2
import time
import numpy as np
from scipy.signal import butter,filtfilt

bin_file_dir = input("Bin file path: ")
num_samples = 256 # int(input("Number of samples per chirp: "))
num_chirps = 128 # int(input("Number of chirps per frame: "))
num_frames = 256 # int(input("Number of frames: "))

## Filter
b, a = butter(2, 0.03, btype='highpass', analog=False)

data = np.fromfile(bin_file_dir, dtype=np.uint16)
data = data.reshape((num_frames, num_chirps, num_samples, 8))[:, :, :, [0, 4]]
beat = (data[:, :, :, 0] + 1j*data[:, :, :, 1]).astype(np.complex64)

radarRange = np.fft.fft(beat, axis=2)
radarRange = np.rot90(radarRange, 1, axes=(1, 2))
radarRange = filtfilt(b, a, radarRange)
doppler = np.abs(np.fft.fft(radarRange))

doppler = np.uint8((doppler-np.min(doppler))/(np.max(doppler)-np.min(doppler))*255)

doppler_map = np.empty_like(doppler)
for frame in range(doppler.shape[0]):
    doppler_map[frame, :, :num_chirps//2], doppler_map[frame, :, num_chirps//2:] = doppler[frame, :, num_chirps//2:], doppler[frame, :, :num_chirps//2]

window_name = 'Range Doppler'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

for i in range(num_frames):
    cv2.imshow(window_name, cv2.applyColorMap(doppler_map[i], cv2.COLORMAP_JET))
    time.sleep(0.1)
    if cv2.waitKey(25) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
