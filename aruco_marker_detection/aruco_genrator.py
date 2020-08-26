from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# parameters
MARKER_COUNT = 4
ARUCO_DICT_TYPE = aruco.DICT_4X4_100
MARKER_DIM = 2048  # in pixel
VIZ_BOOL = False

# init
aruco_dict = aruco.Dictionary_get(ARUCO_DICT_TYPE)


def generate_marker():
    '''
    @brief: Generate Aruco marker and save each marker as pdf and png in data folder
    '''
    for i in range(1, MARKER_COUNT+1):
        img = aruco.drawMarker(aruco_dict, i, MARKER_DIM)
        plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        plt.axis('off')
        plt.savefig("data/marker_{}.pdf".format(i),
                    papertype='a4', orientation='portrait', format='pdf')
        plt.savefig("data/marker_{}.png".format(i))
        if VIZ_BOOL:
            plt.show()


def main():
    generate_marker()


if __name__ == '__main__':
    main()
