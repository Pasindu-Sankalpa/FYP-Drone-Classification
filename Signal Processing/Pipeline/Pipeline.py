import os
import cv2
import time
import json
import mat73
import numpy as np
from matlab import engine
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data_folder = input("Radar data folder to work on: ") # Ex: 'Pipeline\PostProc_22_5'
file_dir = os.path.join(os.getcwd(), data_folder)

## Check if .mat files are available, if not create them using .bin file

if os.path.exists(os.path.join(file_dir, "adcData.mat")) and os.path.exists(os.path.join(file_dir, "radarCube.mat")):
    print("[INFO] .mat files available")
else:
    print("[INFO] Creating .mat files")
    setup_json_dir = os.path.join(file_dir, "SETUP.setup.json")

    with open(setup_json_dir, 'r', encoding='utf-8') as json_file:
        setup_json = json.load(json_file)

    # Update file paths of .json files to working directory
    setup_json["configUsed"] = os.path.join(file_dir, "MMWAVE.mmwave.json")
    setup_json["capturedFiles"]["fileBasePath"] = file_dir

    # Save updated .json file
    with open(setup_json_dir, 'w', encoding='utf-8') as json_file:
        json.dump(setup_json, json_file)

        print("[INFO] setup.json file updated successfully")

    # Create .mat files using MATLAB
    eng = engine.start_matlab()
    path = eng.genpath(os.getcwd()) # Add current directory to MATLAB path
    eng.addpath(path, nargout= 0 )
    try:
        eng.raw_data_reader(setup_json_dir, os.path.join(data_folder, "adcData"), os.path.join(data_folder, "radarCube"), 0, nargout=0)
        eng.verify_data(os.path.join(data_folder, "radarCube.mat"), nargout=0)
    except: print("[Error] Unable to create .mat files")
    else: print("[INFO] .mat files created successfully")
    eng.verify_data(os.path.join(data_folder, "radarCube.mat"), nargout=0)
    eng.quit()

## Process data
radarCube = mat73.loadmat(os.path.join(file_dir, "radarCube.mat"))
radar_range = np.array(radarCube["radarCube"]["data"])[:, :, 0, :]
doppler = np.rot90(np.abs(np.fft.fft(radar_range, axis=1)), 1, axes=(1, 2))
doppler = np.uint8((doppler-np.min(doppler))/(np.max(doppler)-np.min(doppler))*255)

doppler_map = np.empty_like(doppler)
for frame in range(doppler.shape[0]):
    doppler_map[frame, :, :64], doppler_map[frame, :, 64:] = doppler[frame, :, 64:], doppler[frame, :, :64]

## Plot using openCV

window_name = 'Range Doppler'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

for i in range(doppler_map.shape[0]):
    cv2.imshow(window_name, cv2.applyColorMap(doppler_map[i], cv2.COLORMAP_JET))
    time.sleep(0.1)
    if cv2.waitKey(25) & 0xFF == ord('q'): break
cv2.destroyAllWindows()

## Plot using matplotlib (slower)

# def updataFigure(i):
#     global doppler
#     ax.imshow(doppler[i])
#     ax.set_xlabel("Velocity")
#     ax.set_ylabel("Distance")
#     if i == doppler.shape[0]-1: exit()

# fig = plt.figure(figsize=(12,6), facecolor='#DEDEDE', num="Range-Doppler")
# ax = plt.subplot(111)
# ax.set_facecolor('#DEDEDE')

# ani = FuncAnimation(fig, updataFigure, interval=10, cache_frame_data=False, repeat=False)
# plt.show()

print("[INFO] Exit with no error")