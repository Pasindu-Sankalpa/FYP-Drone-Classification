import numpy as np
import cv2
import time

cfar_dopper_map = np.load(input("numpy array file path: "))

window_name = 'Range Doppler'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

for i in range(cfar_dopper_map.shape[0]):
    cv2.imshow(window_name, cfar_dopper_map[i])
    time.sleep(0.1)
    if cv2.waitKey(25) & 0xFF == ord('q'): break
cv2.destroyAllWindows()