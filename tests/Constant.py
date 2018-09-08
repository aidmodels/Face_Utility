import unittest
from face_utility.bundle import Bundle

class BundleTester(unittest.TestCase):
    
    def testInitSolver(self):
        print(Bundle.POSE_PREDICTOR_LOCATION)
        print(Bundle.PRETRAINED_TOML)
        return True

if __name__ == '__main__':
    unittest.main()