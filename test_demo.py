import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from media_pipe_hands import *
from mediapipe_holistic import *
import kalman_filters as kf
from fuse_signals import *
from particle_filters import *


def plot_signal(results, title):
    fig = plt.figure(figsize=(15, 10))
    # for i, result in enumerate(results):
    result = np.array(results[0])
    print(f"{title},{result.shape}")
    frame = np.arange(result.shape[0])
    result_x = result[:, 0]
    result_y = result[:, 1]
    # result_z = result[:, 2]

    plt.suptitle(f"Signal Fusion Results")
    ax = fig.add_subplot(4, 2, 1)
    ax.plot(frame, result_x)
    ax.set_xlabel("frame")
    ax.set_ylabel("MP Hands position x")
    ax = fig.add_subplot(4, 2, 2)
    ax.plot(frame, result_y)
    # ax.set_xlabel("frame")
    ax.set_ylabel("MP Hands position y")


    result = np.array(results[1])
    print(f"{title},{result.shape}")
    frame = np.arange(result.shape[0])
    result_x = result[:, 0]
    result_y = result[:, 1]
    # plt.suptitle(f"Signal MP Pose")
    ax = fig.add_subplot(4, 2,3)
    ax.plot(frame, result_x)
    # ax.set_xlabel("frame")
    ax.set_ylabel("MP Holistic position x")
    ax = fig.add_subplot(4, 2, 4)
    ax.plot(frame, result_y)
    ax.set_xlabel("frame")
    ax.set_ylabel("MP Holistic position y")

    result = np.array(results[2])
    print(f"{title},{result.shape}")
    frame = np.arange(result.shape[0])
    result_x = result[:, 0]
    result_y = result[:, 1]
    # plt.suptitle(f"Signal MP Pose")
    ax = fig.add_subplot(4, 2, 5)
    ax.plot(frame, result_x)
    ax.set_xlabel("frame")
    ax.set_ylabel("Kalman position x")
    ax = fig.add_subplot(4, 2, 6)
    ax.plot(frame, result_y)
    ax.set_xlabel("frame")
    ax.set_ylabel("Kalman position y")

    result = np.array(results[3])
    print(f"{title},{result.shape}")
    frame = np.arange(result.shape[0])
    result_x = result[:, 0]
    result_y = result[:, 1]
    # plt.suptitle(f"Signal MP Pose")
    ax = fig.add_subplot(4, 2, 7)
    ax.plot(frame, result_x)
    ax.set_xlabel("frame")
    ax.set_ylabel("Particle Filter position x")
    ax = fig.add_subplot(4, 2, 8)
    ax.plot(frame, result_y)
    ax.set_xlabel("frame")
    ax.set_ylabel("Particle Filter position y")
    # ax = fig.add_subplot(2, 2, 3)
    # ax.plot(np.array(frame), result_z)
    # ax.set_xlabel("frame")
    # ax.set_ylabel("position in direction z")

    # # plot 3d trajectory of hand wrist
    # ax = fig.add_subplot(2, 2, 4, projection='3d')
    # ax.set_xlabel("z")
    # ax.set_ylabel("x")
    # ax.set_zlabel("y")
    # ax.set_title("Trajectory of Hand wrist point", y=-0.3)
    # ax.plot((np.array(result_z)), np.array(result_x), np.array(result_y),
    #         c='r')
    # plt.savefig("./results/" + title + ".jpg")
    plt.show()


def main():
    # number of video
    number = "5_6"
    path_to_videos = "path/to/videos/"
    cap_1 = cv2.VideoCapture(path_to_videos + "id_" + number + ".mkv")
    # cap.release()
    signal1 = np.array(get_pos_of_joint_from_video(path_to_videos + "id_" + number + ".mkv", 0))

    cap_2 = cv2.VideoCapture(path_to_videos + "id_" + number + ".mkv")
    # cap.release()
    signal2 = np.array(get_hand_landmark_from_video(path_to_videos + "id_" + number + ".mkv"))

    fps = 15
    signals = np.hstack((signal1[:,:2],signal2[:,:2]))

    timestamps = [1 / fps, 1 / fps]
    ini_pos = [signal1[0][0], signal1[0][1], 0, 0]
    fuse_signal = FuseSignal(signals, timestamps)
    filter = ParticleFilter(1000, dt=1 / fps, state_dim=2)

    fused_signal, kalman_filtered = fuse_signal.particle_filter(filter, ini_pos)
    title = "id_" + number + "hand"
    plot_signal([signal1, signal2, kalman_filtered, fused_signal], title)
    # title = "id_" + number + "holistic"
    # plot_signal(signal2, title)
    # title = "id_" + number + "Kalman+Particle_filtered"
    # plot_signal([fused_signal, signal1], title)


main()
