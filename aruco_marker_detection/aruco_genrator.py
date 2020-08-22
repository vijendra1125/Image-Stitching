from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# parameters
marker_count = 4
aruco_dict = aruco.DICT_4X4_100
marker_dim = 2048  # in pixel
viz_bool = False

# init
aruco_dict = aruco.Dictionary_get(aruco_dict)


def generate_marker():
    '''
    @brief: Generate Aruco marker and save each marker as pdf and png in data folder
    '''
    for i in range(1, marker_count+1):
        img = aruco.drawMarker(aruco_dict, i, marker_dim)
        plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        plt.axis('off')
        plt.savefig("data/marker_{}.pdf".format(i),
                    papertype='a4', orientation='portrait', format='pdf')
        plt.savefig("data/marker_{}.png".format(i))
        if viz_bool:
            plt.show()


def main():
    generate_marker()


if __name__ == '__main__':
    main()
