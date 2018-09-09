import unittest
import requests

url = "http://127.0.0.1:8080/infer"

class ServiceTester(unittest.TestCase):
    
    def test_cnn(self):
        file = open("tests/lena.png", "rb")
        files = {'file': file}        
        config = {
            "number_of_times_to_upsample": 1,
            "mode": "CNN"
        }
        r = requests.post(url, files=files, data=config)
        file.close()
        print (r.text)

    def test_hog(self):
        file = open("tests/lena.png", "rb")
        files = {'file': file}
        
        config = {
            "number_of_times_to_upsample": 1,
            "mode": "CNN"
        }
        r = requests.post(url, files=files, data=config)
        file.close()
        print (r.text)

if __name__ == '__main__':
    unittest.main()