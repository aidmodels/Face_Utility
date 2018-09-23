import unittest
from face_utility.solver import FaceDetectionSolver
from face_utility.solver import FaceLandmarkSolver

class DetectionSolverTester(unittest.TestCase):

    def testInit(self):
        self.so = FaceDetectionSolver()
        self.assertTrue(self.so.is_ready)
        
    def testHOGDetect(self):
        self.testInit()
        results = self.so.infer("tests/lena.png", {"mode":"HOG", "number_of_times_to_upsample":1})
        print(results)

    def testCNNDetect(self):
        self.testInit()
        results = self.so.infer("tests/lena.png", {"mode":"CNN", "number_of_times_to_upsample":1})
        print(results)

class LandmarkSolverTester(unittest.TestCase):
    
    def testInit(self):
        self.so = FaceLandmarkSolver()
        self.assertTrue(self.so.is_ready)
    
    def testLandmarkSmall(self):
        self.testInit()
        results = self.so.infer("tests/lena.png", {"mode":"small"})
        print(results)
    
    def testLandmarkLarge(self):
        self.testInit()
        results = self.so.infer("tests/lena,png", {"mode":"large"})
        print(results)
        
if __name__ == '__main__':
    unittest.main()