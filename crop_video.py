import h5py
import argparse
import os


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


if __name__ == "__main__":
    main()