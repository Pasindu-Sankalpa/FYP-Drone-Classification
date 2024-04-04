## import
import os
import cv2
import time
import numpy as np
from scipy.signal import butter,filtfilt
import sys
import random

## https://qiita.com/harmegiddo/items/8a7e1b4b3a899a9e1f0c

def cfar(inputImg):
    # fields
    GUARD_CELLS = 5
    BG_CELLS = 10
    ALPHA = 20
    CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
    HALF_CFAR_UNITS = int(CFAR_UNITS/2) + 1

    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1]), np.uint8)

    # search
    for i in range(inputImg.shape[0] - CFAR_UNITS):
        center_cell_x = i + BG_CELLS + GUARD_CELLS
        for j in range(inputImg.shape[1] - CFAR_UNITS):
            center_cell_y = j  + BG_CELLS + GUARD_CELLS
            average = 0
            for k in range(CFAR_UNITS):
                for l in range(CFAR_UNITS):
                    if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (l < (CFAR_UNITS - BG_CELLS)):
                        continue
                    average += inputImg[i + k, j + l]
            average /= (CFAR_UNITS * CFAR_UNITS) - ( ((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1) )

            if inputImg[center_cell_x, center_cell_y] > (average * ALPHA):
                estimateImg[center_cell_x, center_cell_y] = 255

    return estimateImg

bin_file_dir = input("Bin file path: ")
num_samples = int(input("Number of samples per chirp: ")) # 256
num_chirps = int(input("Number of chirps per frame: ")) # 128
num_frames = int(input("Number of frames: ")) # 256

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

cfar_dopper_map = []

for i in range(num_frames):
    print(i, end =" ")
    cfar_dopper_map.append(cfar(doppler_map[i]))
    print("Done CFAR")

print("Done CFAR")

window_name = 'Range Doppler'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

for i in range(128):
    cv2.imshow(window_name, cv2.applyColorMap(cfar_dopper_map[i], cv2.COLORMAP_JET))
    time.sleep(1)
    if cv2.waitKey(25) & 0xFF == ord('q'): break
cv2.destroyAllWindows()