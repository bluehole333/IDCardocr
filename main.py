import datetime

from ocr import Ocr, IDCard

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    ocr = IDCard('test_img/test2.jpg')
    ocr.get_result()
    endtime = datetime.datetime.now()
    print("运行时间:", (endtime - starttime).seconds)
