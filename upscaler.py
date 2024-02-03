from PIL import Image
import cv2
import numpy as np
from cv2 import dnn_superres
import onnxruntime as ort

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

def SRCNN(source):
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "srcnn.pb"  # Path to your SRCNN model file
    sr.readModel(path)
    sr.setModel("srcnn", 2)  # Set the model type and scale factor
    source = np.array(source)
    result = sr.upsample(source)
    return Image.fromarray(result)
def EDSR(source):
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "EDSR_x4.pb"  # Path to your EDSR model file
    sr.readModel(path)
    sr.setModel("edsr", 4)  # Set the model type and scale factor
    source = np.array(source)
    result = sr.upsample(source)
    return Image.fromarray(result)
    
def BSRGAN(source):
    sess = ort.InferenceSession("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx")
    source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    source = source.astype(np.float32) / 255.0
    source = np.transpose(source, (2, 0, 1))
    source = np.expand_dims(source, 0)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: source})
    output = output[0]
    output = np.squeeze(output)
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))