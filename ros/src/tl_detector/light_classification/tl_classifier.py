from styx_msgs.msg import TrafficLight
from keras.preprocessing.image import img_to_array
import numpy as np
import keras.backend as K
from squeezenet import SqueezeNet
from consts import IMAGE_WIDTH, IMAGE_HEIGHT
import rospy
import cv2
import os
import tensorflow as tf
from keras.models import load_model

# Model vgg16_trafficlight_simulator_model -> https://drive.google.com/open?id=0B5_xbblUg-gDR1FNUmRGekdNRFE

class TLClassifier(object):
    def __init__(self):
        rospy.loginfo("TLClassifier starting")
        K.set_image_dim_ordering('tf')
        # self.model = SqueezeNet(3, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # self.model = load_model(fname)
        fname = os.path.join('light_classification', 'trained_model/vgg16_trafficlight_simulator_model.h5')
        self.model = load_model(fname)
        # self.model.load_weights(fname)
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        rospy.loginfo("TLClassifier get_classification")
        image = cv2.resize(image, (200, 200))
        image = img_to_array(image)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        with self.graph.as_default():
            preds = self.model.predict(image)[0]
        prediction_result = np.argmax(preds)

        if prediction_result == 0:
            rospy.loginfo('tl_classifier: Red traffic light detected.')
            print("Red traffic light")
            return TrafficLight.RED
        elif prediction_result == 2:
            rospy.loginfo('tl_classifier: Green traffic light detected.')
            return TrafficLight.GREEN
        else:
            rospy.loginfo('tl_classifier: Unknown traffic light detected')
            return TrafficLight.UNKNOWN
