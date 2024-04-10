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
    parser.add_argument('--res_x', type=int, default=224, help='width of output video')
    parser.add_argument('--res_y', type=int, default=224, help='height of output video')
    args = parser.parse_args()

    h5_file = h5py.File(args.h5_file, 'r')

    video_key = os.path.splitext((os.path.split(args.video_fn)[1]))[0]
    if video_key not in h5_file.keys():
        print("Videofile {} is not available in h5 file.".format(args.video_fn))
        return

    pose = h5_file[video_key]['joints']['pose_landmarks'][:, :, :2]
    pose = np.reshape(pose, (-1, 2), order='F')

    min_x = np.min(pose[:, 1])
    min_y = np.min(pose[:, 0])

    max_x = np.max(pose[:, 1])
    max_y = np.max(pose[:, 0])

    width = int(max_x - min_x)
    height = int(max_y - min_y)

    rect_size = np.maximum(width, height)
    center = (min_x + width // 2, min_y + height // 2)

    rect_x = int(center[0] - rect_size // 2)
    rect_y = int(center[1] - rect_size // 2)

    video = cv2.VideoCapture(args.video_fn)

    while True:
        ret, im = video.read()

        if not ret:
            break

        cv2.rectangle(im, (rect_x, rect_y), (rect_x + rect_size, rect_y + rect_size), (0, 255, 0), 3)
        cv2.circle(im, (int(center[0]), int(center[1])), 10, (255, 0, 0), -1)
        cv2.imshow("image", im)
        cv2.waitKey(10)

    pass

if __name__ == "__main__":
    main()
