import h5py
import argparse
import os
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(description='crop video',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video_fn', type=str, help='video file')
    parser.add_argument('h5_file', type=str, help='h5 with joints')
    parser.add_argument('output', type=str, help='path to output video')
    parser.add_argument('--resolution', type=int, default=640, help='size of output video (it\'s a squre)')
    parser.add_argument('--border_size', type=float, default=0.3, help='size of added border')
    args = parser.parse_args()

    h5_file = h5py.File(args.h5_file, 'r')

    video_key = os.path.splitext((os.path.split(args.video_fn)[1]))[0]
    if video_key not in h5_file.keys():
        print("Videofile {} is not available in h5 file.".format(args.video_fn))
        return

    pose = h5_file[video_key]['joints']['pose_landmarks'][:, :22, :2]
    left_shoulders = pose[:, 11, :]
    right_shoulders = pose[:, 12, :]
    mean_left_shoulder = np.mean(left_shoulders, axis=0)
    mean_right_shoulder = np.mean(right_shoulders, axis=0)
    shoulder_width = np.linalg.norm(mean_left_shoulder - mean_right_shoulder)
    print(mean_left_shoulder, mean_right_shoulder)
    print(shoulder_width)

    pose = np.reshape(pose, (-1, 2), order='F')

    rect_size = 3 * shoulder_width
    # size of rect is adjusted as percentage
    rect_size *= (1 + args.border_size)
    rect_size = int(rect_size)

    center = np.mean(pose, axis=0)

    rect_x = int(center[0] - rect_size // 2)
    rect_y = int(center[1] - rect_size // 2)

    video = cv2.VideoCapture(args.video_fn)

    while True:
        ret, im = video.read()

        if not ret:
            break

        source_points = np.float32([[rect_x, rect_y], [rect_x + rect_size, rect_y], [rect_x, rect_y + rect_size]])
        target_points = np.float32([[0, 0], [rect_size, 0], [0, rect_size]])
        M = cv2.getAffineTransform(source_points, target_points)
        canvas = cv2.warpAffine(im, M, (rect_size, rect_size))

        canvas = cv2.resize(canvas, (args.resolution, args.resolution))

        cv2.rectangle(im, (rect_x, rect_y), (rect_x + rect_size, rect_y + rect_size), (0, 255, 0), 3)

        for p in pose:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

        cv2.circle(im, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)

        cv2.imshow("image", canvas)
        cv2.waitKey(20)


if __name__ == "__main__":
    main()
