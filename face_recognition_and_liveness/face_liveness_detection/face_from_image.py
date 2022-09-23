import numpy as np
import argparse
import os
import cv2

#xây dựng trình phân tích cú pháp đối số và phân tích cú pháp đối số
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Path to input image')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Path to output directory of cropped face')
parser.add_argument('-d', '--detector', type=str, required=True,
                    help='Path to OpenCV\'s face detector')
parser.add_argument('-c', '--confidence', type=int, default= 0.5 ,
                    help='Confidence of face detection')
args = vars(parser.parse_args())

print('[INFO] loading face detector')
proto_path = os.path.sep.join([args['detector'],'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'],
                               'res10_300x300_ssd_iter_140000.caffemodel'])

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# đọc ảnh
image = cv2.imread(args['input'])

# tên ảnh
if len(os.listdir(args['output'])) > 0:
    latest_file = 0
    for file in os.listdir(args['output']):
        latest_file = max(latest_file, int(file[:file.find('.')]))
    #để nó không thay thế hình ảnh mới nhất trong thư mục
    latest_file += 1
else:
    latest_file = 0
    
saved_name = latest_file

# tạo một đốm màu từ hình ảnh (hình ảnh tiền xử lý)
# về cơ bản, nó có nghĩa là trừ và chia tỷ lệ
# (104.0, 177.0, 123.0) là trung bình của hình ảnh trong FaceNet
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0,
                             (300,300), (104.0, 177.0, 123.0))

#vượt qua đốm màu thông qua NN và nhận được các phát hiện
net.setInput(blob)
detections = net.forward()

# đảm bảo ít nhất 1 khuôn mặt mà nó phát hiện được
if len(detections) > 0:
    # chúng tôi đang đưa ra giả định rằng mỗi hình ảnh CHỈ CÓ MỘT khuôn mặt,
    # để tìm hộp giới hạn có xác suất lớn nhất
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    # đảm bảo rằng việc phát hiện với xác suất cao nhất
    # vượt qua ngưỡng xác suất tối thiểu của chúng tôi (giúp lọc ra một số phát hiện yếu)
    if confidence > args['confidence']:
        # tính tọa độ (x, y) của hộp giới hạn
        # cho khuôn mặt và trích xuất ROI trên khuôn mặt
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        face = image[startY:endY, startX:endX]

        # ghi hình ảnh vào đĩa
        p = os.path.sep.join([args['output'], f'{saved_name}.png'])
        cv2.imwrite(p, face)
        print(f'[INFO] saved {p} to disk')
        
