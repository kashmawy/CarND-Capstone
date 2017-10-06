
import os
import shutil

import argparse
import rosbag
import cv_bridge
import cv2
import tqdm


def main(bag_path, output_path):

    shutil.rmtree(output_path, ignore_errors=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bridge = cv_bridge.CvBridge()


    with rosbag.Bag(bag_path) as bag:

        messages = list(bag.read_messages(topics="/image_raw"))

        padding = len(str(len(messages))) + 1

        for index, message in enumerate(tqdm.tqdm(messages)):

            data = bridge.imgmsg_to_cv2(message.message, "bgr8")

            file_name = str(index).zfill(padding) + ".jpg"
            path = os.path.join(output_path, file_name)
            cv2.imwrite(path, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", help="bag path to extract")
    parser.add_argument("--output_path", help="bag path to extract")
    args = parser.parse_args()

    print "Extracting from {} to {}".format(args.bag_path, args.output_path)

    main(args.bag_path, args.output_path)
