import requests

url = "http://127.0.0.1:8080/infer"

def test_encode():
    file = open("tests/lena.png", "rb")
    files = {'file': file}        
    config = {
        "delete_after_process": "false",
        "num_jitters":"1",
    }
    r = requests.post(url, files=files, data=config)
    file.close()
    print (r.text)

test_encode()