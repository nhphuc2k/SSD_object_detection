import cv2

confThreshold = 0.5
nmsThreshold = 0.3
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n') #mo file coco.names va nhan mau ngau nhien

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
# cung cấp các tệp nhận config và weight
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold,nmsThreshold)
   # print(classIds,bbox)
    if len(objects) == 0:
        objects = classNames
        objectInfo=[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2) #vẽ khung hình chữ nhật bao quanh object
                    cv2.putText(img,className.upper(),(box[0]+10,box[1]+30), #viết tên của object
                    cv2.FONT_HERSHEY_COMPLEX_SMALL ,1,(255,0,0),2) # font chữ
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30), #hiện tỉ lệ phần trăm nhận dạng object
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)

    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture('123.mp4')
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 70)

while True:
    success, img = cap.read()
    result,objectInfo = getObjects(img)
    print(objectInfo)
    cv2.imshow("CAMERA", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()