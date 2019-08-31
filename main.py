from sift import find_idcard, ocr

if __name__ == "__main__":
    id_card = find_idcard('test_img/test.jpg')

    if id_card:
        print("没有检测到身份证信息")
    else:
        print(ocr(id_card))
