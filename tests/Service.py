import unittest
import requests

url = "http://127.0.0.1:8080/infer"

class ServiceTester(unittest.TestCase):
    
    def test_cnn(self):
        files = {'file': open("tests/lena.png", "rb")}
        config = {
            "number_of_times_to_upsample": 1,
            "mode": "cnn"
        }
        r = requests.post(url, files=files, data=config)
        print (r.text)



if __name__ == '__main__':
    unittest.main()