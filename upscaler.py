from PIL import Image
import cv2
import numpy as np
from cv2 import dnn_superres

def txsal(source):
    width, height = source.size
    new_width = width * 2
    new_height = height * 2
    sally = Image.new('RGB', (new_width, new_height))
    
    for y in range(height):
        for x in range(width):
            color = source.getpixel((x, y))
            sally.putpixel((x * 2, y * 2), color)
            sally.putpixel((x * 2 + 1, y * 2), color)
            sally.putpixel((x * 2, y * 2 + 1), color)
            sally.putpixel((x * 2 + 1, y * 2 + 1), color)
    return sally

def area(source):
    width, height = source.size
    new_width = width * 2
    new_height = height * 2
    source = np.array(source)
    ariel = cv2.resize(source, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(ariel)

def lanczos(source):
    width, height = source.size
    new_width = width * 2
    new_height = height * 2
    source = np.array(source)
    lance = cv2.resize(source, dsize=(new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(lance)

def cubic(source):
    width, height = source.size
    new_width = width * 2
    new_height = height * 2
    source = np.array(source)
    buick = cv2.resize(source, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(buick)

def FSRCNN(source):
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "FSRCNN_x2.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn", 2)
    source = np.array(source)
    result = sr.upsample(source)
    return Image.fromarray(result)

def ESPCN(source):
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "ESPCN_x2.pb"
    sr.readModel(path)
    sr.setModel("espcn", 2)
    source = np.array(source)
    result = sr.upsample(source)
    return Image.fromarray(result)

def SRN(source):
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "LapSRN_x2.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 2)
    source = np.array(source)
    result = sr.upsample(source)
    return Image.fromarray(result)