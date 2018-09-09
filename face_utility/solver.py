import logging
import dlib
from cvpm.solver import Solver
from cvpm.utility import load_image_file
from face_utility.bundle import FaceUtilityBundle as Bundle

class FaceDetectionSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
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