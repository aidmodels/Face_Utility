from pkg_resources import resource_filename
from cvpm.bundle import Bundle
class FaceUtilityBundle(Bundle):
    PRETRAINED_TOML = resource_filename(__name__, "../pretrained/pretrained.toml")
    POSE_PREDICTOR_LOCATION = resource_filename(__name__, "../pretrained/shape_predictor_68_face_landmarks.dat")
    CNN_FACE_DETECTOR_LOCATION = resource_filename(__name__, "../pretrained/mmod_human_face_detector.dat")
    POSE_PREDICTOR_FIVE_LOCATION = resource_filename(__name__, "../pretrained/shape_predictor_5_face_landmarks.dat")
    FACE_RECOGNITION_LOCATION = resource_filename(__name__, "../pretrained/dlib_face_recognition_resnet_model_v1.dat")
    ENABLE_TRAIN = False
    SOLVERS = []