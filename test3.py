import numpy as np
import cv2

import torchreid
import torch
from PIL import Image
from torchreid.reid.utils import FeatureExtractor
import argparse
from PIL import Image,ImageDraw,ImageFont
# Initialize the feature extractor with a pretrained model
extractor = FeatureExtractor(
    model_name='osnet_x1_0',  # You can choose other models too, like 'resnet50', 'resnet101'
    model_path='osnet_x1_0_imagenet.pth',  # None to use default pretrained model
    # device='cuda'  # or 'cpu'
    device='cuda'  # or 'cpu'
)

def extract_feature(image_path):
    # 直接传递图像路径给 FeatureExtractor
    features = extractor(image_path)  # Extract features
    return features[0]  # Return the feature vector

def compare_images(image_path1, image_path2):
    # Extract features from both images
    features1 = extract_feature(image_path1)
    features2 = extract_feature(image_path2)

    # Compute the cosine similarity
    similarity = torch.nn.functional.cosine_similarity(features1, features2, dim=0)
    return similarity.item()

def comparetwoimg(image1,image2):
   
    
    similarity_score = compare_images(image1, image2)

    threshold = 0.7  # Adjust this threshold based on your use case
    if similarity_score > threshold:
        print("The images are likely of the same person."+"likelihood:"+str(similarity_score))
    else:
        print("The images are likely of different people."+"likelihood:"+str(similarity_score))

    return similarity_score
def showChinesePutText(img,str,x,y,Scale):
     
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    pilimg = Image.fromarray(cv2img)
# # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg) # 图片上打印
# #simsun 宋体
    font = ImageFont.truetype(r"simsun.ttc", 40) 
# #位置，文字，颜色==红色，字体引入
    draw.text((x, y), str, Scale, font=font) 
# # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    return cv2charimg
def showcapture(): 
    low_confidence=0.8
    proto_txt=r'D:\python\deploy.prototxt'
    model_path=r'D:\python\res10_300x300_ssd_iter_140000_fp16.caffemodel'
 
    # 加载模型
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(proto_txt, model_path)
 
    capture = cv2.VideoCapture(0) # 打开笔记本内置摄像头
    while (capture.isOpened()): # 笔记本内置摄像头被打开后
        retval, image = capture.read() # 从摄像头中实时读取视频
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
    
        # 通过网络传递blob并获得检测和预测
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
    
        # 循环检测
        # TODO(You):请实现循环检测的代码。
        # 循环检测
        laststart=200
        lasty=200
        for i in range(0, detections.shape[2]):
            # 提取与相关的置信度（即概率）
            # 预测
            confidence = detections[0, 0, i, 2]
            # 通过确保“置信度”来过滤掉弱检测
            # 大于最小置信度
            if confidence > low_confidence:
                # 计算边界框的 (x, y) 坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # 绘制人脸的边界框以及概率
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
               
                laststart=startX
                lasty=y+50
                dictname={}
                imageall= [r'D:\python\某某某.jpg',r'D:\python\汪汪汪.jpg',r'D:\python\哈哈.jpg',r'D:\python\嘟嘟.jpg']
               
                for _,image2 in enumerate(imageall):
                    curfilename=image2.split(sep='\\')[-1]
                    curfilename=curfilename.split(sep='.')[0]
                    img_pil=Image.open(image2)
                    temparray=np.array(img_pil)
                    dictname[curfilename]=comparetwoimg(image,temparray)

                text=max(dictname,key=dictname.get)
                image=showChinesePutText(image,text,startX,y,(0, 0, 255))
                #cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("Video", image) # 在窗口中显示读取到的视频
        key = cv2.waitKey(1) # 窗口的图像刷新时间为1毫秒
        try:
            if cv2.getWindowProperty("Video", cv2.WND_PROP_AUTOSIZE) < 1:
             break
        except:
            break
       
        if key == 32: # 如果按下空格键
            break
    capture.release() # 关闭笔记本内置摄像头
    cv2.destroyAllWindows()  
def main():
    low_confidence=0.8
    image_path='2.jpg'
    proto_txt='deploy.prototxt'
    model_path='res10_300x300_ssd_iter_140000_fp16.caffemodel'
 
    # 加载模型
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(proto_txt, model_path)
 
    # 加载输入图像并为图像构建一个输入 blob
    # 将大小调整为固定的 300x300 像素，然后对其进行标准化
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
 
    # 通过网络传递blob并获得检测和预测
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
 
    # 循环检测
    # TODO(You):请实现循环检测的代码。
    # 循环检测
    laststart=200
    lasty=200
    for i in range(0, detections.shape[2]):
        # 提取与相关的置信度（即概率）
        # 预测
        confidence = detections[0, 0, i, 2]
        # 通过确保“置信度”来过滤掉弱检测
        # 大于最小置信度
        if confidence > low_confidence:
            # 计算边界框的 (x, y) 坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # 绘制人脸的边界框以及概率
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            laststart=startX
            lasty=y+50
    text="beauty"
    cv2.putText(image,text,(laststart,lasty),cv2.FONT_HERSHEY_COMPLEX,0.8,(0, 255, 0),1)
 
    # 展示图片并保存
    cv2.imshow("Output", image)
    cv2.imwrite("01.jpg",image)
    cv2.waitKey(0)
if __name__=='__main__':
    #image = cv2.imread(r'D:\python\1.jpg')
    
    showcapture()
    #comparetwoimg(image,image2)#showcapture()