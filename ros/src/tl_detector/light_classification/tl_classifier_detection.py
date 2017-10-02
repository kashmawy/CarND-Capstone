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

CHUNK_SIZE = 1024

class TLClassifierDetection(object):
    def __init__(self, model_dir):
        rospy.loginfo("TLClassifier Detection starting")
        K.set_image_dim_ordering('tf')


        # self.model = SqueezeNet(3, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # fname = os.path.join('light_classification', 'trained_model/challenge1.weights')
        # self.model.load_weights(fname)
        #
        # self.graph = tf.get_default_graph()

        self.model_dir = model_dir
        self.predict_ready = False

        self.tf_session = None
        self.tf_graph = None
        self.config = None

        # was model and weights made whole?
        if not os.path.exists(self.model_dir + '/frozen_inference_graph.pb'):
            # if not - build it back up
            if os.path.exists(self.model_dir + '/parts'):
                output = open(self.model_dir + '/frozen_inference_graph.pb', 'wb')
                chunks = os.listdir(self.model_dir + '/parts')
                chunks.sort()
                for filename in chunks:
                    filepath = os.path.join(self.model_dir + '/parts', filename)
                    with open(filepath, 'rb') as fileobj:
                        for chunk in iter(partial(fileobj.read, CHUNK_SIZE), ''):
                            output.write(chunk)
                output.close()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # set up tensorflow and traffic light classifier
        if self.tf_session is None:
            # get the traffic light classifier
            self.config = tf.ConfigProto(log_device_placement=True)
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.3  # don't hog all the VRAM!
            self.config.operation_timeout_in_ms = 20000 # terminate anything that don't return in 50 seconds
            self.tf_graph = tf.Graph()
            with self.tf_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path+'/frozen_inference_graph.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    self.tf_session = tf.Session(graph=self.tf_graph, config=self.config)
                    # Definite input and output Tensors for self.tf_graph
                    self.image_tensor = self.tf_graph.get_tensor_by_name('image_tensor:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    self.detection_scores = self.tf_graph.get_tensor_by_name('detection_scores:0')
                    self.detection_classes = self.tf_graph.get_tensor_by_name('detection_classes:0')
                    self.num_detections = self.tf_graph.get_tensor_by_name('num_detections:0')
                    self.predict_ready = True


        prediction = TrafficLight.UNKNOWN

        if self.predict_ready:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_expanded = np.expand_dims(image, axis=0)

            # Actual detection
            (scores, classes, num) = self.tf_session.run(
                [self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # calculate prediction
            c = 5
            predict = self.clabels[c]
            cc = classes[0]
            confidence = scores[0]
            if cc > 0 and cc < 4 and confidence is not None and confidence > THRESHOLD:
                c = cc
                predict = self.clabels[c]



        # rospy.loginfo("TLClassifier get_classification")
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        with self.graph.as_default():
            preds = self.model.predict(image)[0]
        prediction_result = np.argmax(preds)

        if prediction_result == 0:
            # rospy.loginfo('tl_classifier: No traffic light detected.')
            return TrafficLight.UNKNOWN
        elif prediction_result == 1:
            # rospy.loginfo('tl_classifier: Red traffic light detected.')
            return TrafficLight.RED
        else:
            # rospy.loginfo('tl_classifier: Green traffic light detected')
            return TrafficLight.GREEN
