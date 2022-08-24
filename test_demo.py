import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from media_pipe_hands import *
from mediapipe_holistic import *
import kalman_filters as kf
from fuse_signals import *
from particle_filters import *


def plot_signal(result, title):
    result = np.array(result)
    print(f"{title},{result.shape}")
    frame = np.arange(result.shape[0])
    result_x = result[:, 0]
    result_y = result[:, 1]
    result_z = result[:, 2]
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle(title)
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(frame, result_x)
    ax.set_xlabel("frame")
    ax.set_ylabel("position in direction x")
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(frame, result_y)
    ax.set_xlabel("frame")
    ax.set_ylabel("position in direction y")
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(np.array(frame), result_z)
    ax.set_xlabel("frame")
    ax.set_ylabel("position in direction z")

    # plot 3d trajectory of hand wrist
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("Trajectory of Hand wrist point", y=-0.3)
    ax.plot((np.array(result_z)), np.array(result_x), np.array(result_y),
            c='r')
    plt.savefig("./results/" + title + ".jpg")
    plt.show()


def main():
    # number of video
    number = "4_3"
    cap = cv2.VideoCapture("./kinect/id_" + number + ".mkv")
    signal1 = get_pos_of_joint_from_video(cap, 0)
    cap = cv2.VideoCapture("./kinect/id_" + number + ".mkv")
    signal2 = get_hand_landmark_from_video(cap)

    signals = [signal1, signal2]
    timestamps = [1 / 30, 1 / 30]
    ini_pos = [signal1[0][0], signal1[0][1], signal1[0][2], 0, 0, 0]
    fuse_signal = FuseSignal(signals, timestamps)
    filter = ParticleFilter(200, dt=1 / 30)
    fused_signal = fuse_signal.particle_filter(filter, ini_pos)
    title = "id_" + number + "hand"
    plot_signal(signal1, title)
    title = "id_" + number + "holistic"
    plot_signal(signal2, title)
    title = "id_" + number + "filtered"
    plot_signal(fused_signal, title)


main()
