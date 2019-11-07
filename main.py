import datetime

from ocr import Ocr

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    ocr = Ocr('test_img/test2.jpg')
    ocr.go()
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
