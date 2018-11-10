import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import cv2
from pylab import savefig

seqName = 'dslr_dance1'
num_frame = 360
connMat = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])


if __name__ == '__main__':
    for i in range(num_frame):
        dataFile = 'output/dslr_dance1/{:04d}.json' .format(i)
        with open(dataFile) as f:
            joints3d = np.array(json.load(f))[:-1, :]   # discard the last joint

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        x = joints3d[:, 0]
        y = -joints3d[:, 2]
        z = joints3d[:, 1]
        ax.scatter(x, y, z, c='r')
        # ax.scatter(joints3d[:, 0], joints3d[:, 1], joints3d[:, 2], c='r')
        for conn in connMat:
            # ax.plot(joints3d[conn, 0], joints3d[conn, 1], joints3d[conn, 2], c='r')
            ax.plot(x[conn], y[conn], z[conn], c='r')
        ax.set_xlim([144 - 192, 144 + 192])
        # ax.set_ylim([0, 384])
        # ax.set_zlim([-192, 192])
        ax.set_ylim([-192, 192])
        ax.set_zlim([0, 384])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.view_init(elev=-162, azim=-147)
        plt.tight_layout()

        savefig('output/dslr_dance1/side_{:04d}.png'.format(i))
        # plt.show()
        plt.clf()
