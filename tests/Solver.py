import unittest
from face_utility.solver import FaceDetectionSolver

class SolverTester(unittest.TestCase):

    def testInit(self):
        self.so = FaceDetectionSolver()
        self.assertTrue(self.so.is_ready)
        
    def testHOGDetect(self):
        self.testInit()
        results = self.so.infer("tests/lena.png", {"mode":"HOG", "number_of_times_to_upsample":"1"})
        print(results)

    def testCNNDetect(self):
        self.testInit()
        results = self.so.infer("tests/lena.png", {"mode":"CNN", "number_of_times_to_upsample":"1"})
        print(results)

if __name__ == '__main__':
    unittest.main()