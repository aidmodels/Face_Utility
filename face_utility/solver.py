import logging
import dlib
import numpy as np
from cvpm.solver import Solver
from cvpm.utility import load_image_file
from face_utility.bundle import FaceUtilityBundle as Bundle

class FaceDetectionSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(Bundle.PRETRAINED_TOML)
        self.logger = logging.getLogger('face_utility')
        self.face_detector = dlib.get_frontal_face_detector()
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(Bundle.CNN_FACE_DETECTOR_LOCATION)
        self.set_bundle(Bundle)
        self._enable_train = Bundle.ENABLE_TRAIN
        self.set_ready()

    def infer(self, image_file, config):
        image_np = load_image_file(image_file)
        if config["mode"]=="CNN":
            raw_results = self.cnn_face_detector(image_np, int(config["number_of_times_to_upsample"]))
            return [self._trim_css_to_bounds(self._rect_to_css(face.rect), image_np.shape) for face in raw_results]
        elif config["mode"] == "HOG":
            raw_results = self.face_detector(image_np, int(config["number_of_times_to_upsample"]))
            return [self._trim_css_to_bounds(self._rect_to_css(face), image_np.shape) for face in raw_results]
        else:
            self.logger.error("[FACE UTILITY] Only HOG and CNN are supported mode yet!")

    def _rect_to_css(self, rect):
        return rect.top(), rect.right(), rect.bottom(), rect.left()
    
    def _trim_css_to_bounds(self, css, image_shape):
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)
 
class FaceLandmarkSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(Bundle.PRETRAINED_TOML)
        self.logger = logging.getLogger('face_utility')
        self.pose_estimator_68 = dlib.shape_predictor(Bundle.POSE_PREDICTOR_LOCATION)
        self.pose_estimator_5 = dlib.shape_predictor(Bundle.POSE_PREDICTOR_FIVE_LOCATION)
        self.face_detector = dlib.get_frontal_face_detector()
        self.detection_config = {
            "mode": "HOG",
            "number_of_times_to_upsample": "1"
        }
        self.set_bundle(Bundle)
        self._enable_train = Bundle.ENABLE_TRAIN
        self.set_ready()
        
    def infer(self, image_file, config):
        landmarks = []
        pose_predictor = None
        image_np = load_image_file(image_file)
        face_locations = self.face_detector(image_np, int(self.detection_config["number_of_times_to_upsample"]))
        if config["mode"] == "small":
            pose_predictor = self.pose_estimator_5
        else:
            pose_predictor = self.pose_estimator_68
        landmarks = [pose_predictor(image_np, face_location) for face_location in face_locations]
        landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
        if config["mode"] == "large":
            return [{
                "chin": points[0:17],
                "left_eyebrow": points[17:22],
                "right_eyebrow": points[22:27],
                "nose_bridge": points[27:31],
                "nose_tip": points[31:36],
                "left_eye": points[36:42],
                "right_eye": points[42:48],
                "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
                "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
            } for points in landmarks_as_tuples]
        elif config["mode"] == "small":
            return [{
                "nose_tip": [points[4]],
                "left_eye": points[2:4],
                "right_eye": points[0:2],
            } for points in landmarks_as_tuples]
        else:
            self.logger.error("[FACE UTILITY] Only small and large are supported mode yet!")            

class FaceEncodingSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(Bundle.PRETRAINED_TOML)
        self.logger = logging.getLogger('face_utility')
        self.pose_estimator_5 = dlib.shape_predictor(Bundle.POSE_PREDICTOR_FIVE_LOCATION)
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_encoder = dlib.face_recognition_model_v1(Bundle.FACE_RECOGNITION_LOCATION)
        self.detection_config = {
            "mode": "HOG",
            "number_of_times_to_upsample": "1"
        }
        self.set_bundle(Bundle)
        self._enable_train = Bundle.ENABLE_TRAIN
        self.set_ready()
    def infer(self, image_file, config):
        image_np = load_image_file(image_file)
        face_locations = self.face_detector(image_np, int(self.detection_config["number_of_times_to_upsample"]))
        landmarks = [self.pose_estimator_5(image_np, face_location) for face_location in face_locations]
        return [np.array(self.face_encoder.compute_face_descriptor(image_np, raw_landmark, int(config['num_jitters']))) for raw_landmark in landmarks]