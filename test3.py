import numpy as np
import cv2
import torch.nn.functional
import face_recognition
from datetime import datetime
from PIL import Image,ImageDraw,ImageFont

last_time=datetime.now()
familynamelist=[]
familyimgrglist=[] 

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
""" def match_faces(self):
    faces = fr.face_locations(self.img)
    face_encodings = fr.face_encodings(self.img, faces)
    self.load_known_faces()
    pil_img = Image.fromarray(self.img)
    draw = ImageDraw.Draw(pil_img)
    for (top, right, bottom, left), cur_encoding in zip(faces, face_encodings):
        # matches = fr.compare_faces(self.encoding_list, cur_encoding)    
        name = 'unknown'
        # 计算已知人脸和未知人脸特征向量的距离，距离越小表示两张人脸为同一个人的可能性越大
        distances = fr.face_distance(self.encoding_list, cur_encoding)
        match_index = np.argmin(distances)
        if matches[match_index]:
            name = self.name_list[match_index]
        # 绘制匹配到的人脸信息
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
        text_width, text_height = draw.textsize(name)
        font = ImageFont.truetype('arial.ttf', 20)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 255, 0))
        draw.text((left + 5, bottom - text_height - 10), name, fill=(255, 255, 255, 255), font=font)
    del draw
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show()
     """
def allfamilyimg():
    imageall= [r'D:\python\姓名一.jpg',r'D:\python\姓名二.jpg',r'D:\python\姓名三.jpg',r'D:\python\姓名四.jpg']
    listname=[]
    listarray=[]

    for _,image2 in enumerate(imageall):
                    curfilename=image2.split(sep='\\')[-1]
                    curfilename=curfilename.split(sep='.')[0]
                    img_pil=Image.open(image2)
                    temparray=np.array(img_pil)
                    listname.append(curfilename)
                    face_locations1 = face_recognition.face_locations(temparray)
                    if len(face_locations1) > 0 :
                     face_encodings1 = face_recognition.face_encodings(temparray, face_locations1)

                    listarray+=face_encodings1
    return listname,listarray
         
    imageczm = face_recognition.load_image_file(r"d:\python\chai zhi ming.jpg")
    imagedll = face_recognition.load_image_file(r"d:\python\dai lan lan.jpg")
    imagecs = face_recognition.load_image_file(r"d:\python\chai si.jpg")
    imagecy = face_recognition.load_image_file(r"d:\python\chai yi.jpg")
def match_faces(img):
    global   familynamelist
    global   familyimgrglist
    faces = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, faces)
    
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    pil_img = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pil_img)
    for (top, right, bottom, left), cur_encoding in zip(faces, face_encodings):
        # matches = fr.compare_faces(self.encoding_list, cur_encoding)    
        name = '未知人士'
        # 计算已知人脸和未知人脸特征向量的距离，距离越小表示两张人脸为同一个人的可能性越大
        distances = face_recognition.face_distance(familyimgrglist, cur_encoding)
        match_index = np.argmin(distances)
        #print (str(distances))
        if distances[match_index]<0.6:
            name = familynamelist[match_index]
        # 绘制匹配到的人脸信息
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
        
        font = ImageFont.truetype(r"simsun.ttc", 40) 
        left1, top1, right1, bottom1  = draw.textbbox((0, 0),name,font=font)
        width1, text_height = right1 - left1, bottom1 - top1
        #draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(0, 255, 0))
        draw.text((left + 5, top + text_height-30), name,(0, 0, 255), font=font)
    
    del draw
    cv2charimg = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return cv2charimg
    
def showcapture(): 
    #low_confidence=0.8
    #proto_txt=r'D:\python\deploy.prototxt'y
    #model_path=r'D:\python\res10_300x300_ssd_iter_140000_fp16.caffemodel'

    global   familynamelist
    global   familyimgrglist
    global   last_time
    familynamelist,familyimgrglist=allfamilyimg()

    # 加载模型
    #print("[INFO] loading model...")
    #net = cv2.dnn.readNetFromCaffe(proto_txt, model_path)
 
    capture = cv2.VideoCapture(0) # 打开笔记本内置摄像头
    while (capture.isOpened()): # 笔记本内置摄像头被打开后
        retval, image = capture.read() # 从摄像头中实时读取视频
        now_time = datetime.now()
        if((now_time-last_time).total_seconds() * 1000>100):
         image=match_faces(image)
         last_time=now_time
        #else:
           # print (str(now_time)+" lasttime:"+str(last_time))
        # (h, w) = image.shape[:2]
        # blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        #     (300, 300), (104.0, 177.0, 123.0))
    
        # # 通过网络传递blob并获得检测和预测
        # print("[INFO] computing object detections...")
        # net.setInput(blob)
        # detections = net.forward()
    
        # # 循环检测
        # # TODO(You):请实现循环检测的代码。
        # # 循环检测
        # laststart=200
        # lasty=200
        # for i in range(0, detections.shape[2]):
        #     # 提取与相关的置信度（即概率）
        #     # 预测
        #     confidence = detections[0, 0, i, 2]
        #     # 通过确保“置信度”来过滤掉弱检测
        #     # 大于最小置信度
        #     if confidence > low_confidence:
        #         # 计算边界框的 (x, y) 坐标
        #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #         (startX, startY, endX, endY) = box.astype("int")
        #         # 绘制人脸的边界框以及概率
        #         text = "{:.2f}%".format(confidence * 100)
        #         y = startY - 10 if startY - 10 > 10 else startY + 10
        #         cv2.rectangle(image, (startX, startY), (endX, endY),
        #                     (0, 0, 255), 2)
               
        #         laststart=startX
        #         lasty=y+50
        #         # dictname={}
        #         # imageall= [r'D:\python\某某某.jpg',r'D:\python\汪汪汪.jpg',r'D:\python\哈哈.jpg',r'D:\python\嘟嘟.jpg']
               
        #         # for _,image2 in enumerate(imageall):
        #         #     curfilename=image2.split(sep='\\')[-1]
        #         #     curfilename=curfilename.split(sep='.')[0]
        #         #     img_pil=Image.open(image2)
        #         #     temparray=np.array(img_pil)
        #            # dictname[curfilename]=comparetwoimg(image,temparray)

        #         #text=max(dictname,key=dictname.get)
        #         image=match_faces(image)#showChinesePutText(image,text,startX,y,(0, 0, 255))
        #         #cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("Video", image) # 在窗口中显示读取到的视频
        key = cv2.waitKey(10) # 窗口的图像刷新时间为1毫秒
        try:
            if cv2.getWindowProperty("Video", cv2.WND_PROP_AUTOSIZE) < 1:
             break
        except:
            break
       
        if key == 32: # 如果按下空格键
            break
    capture.release() # 关闭笔记本内置摄像头
    cv2.destroyAllWindows()  

def judgewhichperson():
    imageczm = face_recognition.load_image_file(r"d:\python\chai zhi ming.jpg")
    imagedll = face_recognition.load_image_file(r"d:\python\dai lan lan.jpg")
    imagecs = face_recognition.load_image_file(r"d:\python\chai si.jpg")
    imagecy = face_recognition.load_image_file(r"d:\python\chai yi.jpg")
    
    image3 = face_recognition.load_image_file(r"d:\python\2.jpg")
    
    # 检测两张照片中的所有人脸
    face_locations1 = face_recognition.face_locations(imageczm)
    face_locations2 = face_recognition.face_locations(imagedll)
    face_locations3 = face_recognition.face_locations(imagecs)
    face_locations4 = face_recognition.face_locations(imagecy)
    
    #face_locations1=face_locations1+face_locations2
    face_locations5 = face_recognition.face_locations(image3)
    
    # 如果至少检测到一张脸
    if len(face_locations1) > 0 and len(face_locations3) > 0:
        # 获取两张照片中人脸的特征
        face_encodings1 = face_recognition.face_encodings(imageczm, face_locations1)
        face_encodings2 = face_recognition.face_encodings(imagedll, face_locations2)
        face_encodings3 = face_recognition.face_encodings(imagecs, face_locations3)
        face_encodings4 = face_recognition.face_encodings(imagecy, face_locations4)

        face_encodings5 = face_recognition.face_encodings(image3, face_locations5)

        face_encodings1=face_encodings1+face_encodings2+face_encodings3+face_encodings4

    
        # 比较两张照片中的人脸特征
        #for face_encoding1 in face_encodings1:
        for face_encoding5 in face_encodings5:
                distances = face_recognition.face_distance(face_encodings3, face_encoding5)
                match_index = np.argmin(distances)
                print(str(distances[match_index]))
                if distances[match_index]<0.4:
                    print("两张照片是同一个人")
                    #break
                else:
                    print("两张照片不是同一个人")
        
    else:
        print("没有检测到脸部")
            
def main():
    low_confidence=0.8
    image_path='5.jpg'
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
    #judgewhichperson()
    showcapture()
    #comparetwoimg(image,image2)#showcapture()