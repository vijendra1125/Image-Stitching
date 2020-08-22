from cv2 import aruco
import cv2
import matplotlib.pyplot as plt

# parameter
viz_bool = True
marker_dict = aruco.DICT_4X4_100

# init
cam = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(marker_dict)
parameters = aruco.DetectorParameters_create()


def detect_image(image, cam_bool=False):
    '''
    @brief: Detect marker in a given image and visualize if viz_bool is True
    @arg[in]: 
        image: RGB image
        cam_bool: True if image is passed from camera feed
    arg[out]: 
        key(if cam bool is True): key pressed during vizualization
    '''
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        image_gray, aruco_dict, parameters=parameters)
    # visualization
    if viz_bool:
        for i in range(len(corners)):
            print('ID:', ids[i], '\ncorners:\n', corners[i])
        frame_markers = aruco.drawDetectedMarkers(
            image.copy(), corners, ids, borderColor=(255, 255, 0))
        cv2.imshow('marker detection', frame_markers)
        if cam_bool:
            key = cv2.waitKey(1)
            return key
        cv2.waitKey(0)


def detect_cam():
    '''
    @brief: Detect marker in camera feed and visualize if viz_bool is True
    '''
    while True:
        # marker detection
        ret, image = cam.read()
        key = detect_image(image, True)
        if key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    # image = cv2.imread('data/marker_1.png')
    # detect_image(image)
    detect_cam()


if __name__ == '__main__':
    main()
