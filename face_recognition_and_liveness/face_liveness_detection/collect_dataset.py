import numpy as np
import argparse
import cv2
import os

# xây dựng trình phân tích cú pháp đối số và phân tích cú pháp đối số
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, 
                    help='Path to input video')
parser.add_argument('-o', '--output', type=str, required=True, 
                    help='Path to output directory of cropped face images')
parser.add_argument('-d', '--detector', type=str, required=True, 
                    help='Path to OpenCV\'s deep learning face detector')
parser.add_argument('-c', '--confidence', type=float, default=0.5, 
                    help='Confidence of face detection')
parser.add_argument('-s', '--skip', type=int, default=16,
                    help='# of frames to skip before applying face detection and crop')
args = vars(parser.parse_args())

# tải máy dò khuôn mặt được tuần tự hóa của chúng tôi từ đĩa
print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'],
                               'res10_300x300_ssd_iter_140000.caffemodel'])
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# mở một con trỏ tới luồng tệp video
# khởi tạo tổng số khung hình đã đọc và lưu
vs = cv2.VideoCapture(args['input'])
read = 0

# trong trường hợp của, đã có một số hình ảnh trong thư mục
if len(os.listdir(args['output'])) > 0:
    latest_file = 0
    for file in os.listdir(args['output']):
        latest_file = max(latest_file,int(file[:file.find('.')]))
    # +1 để nó không thay thế hình ảnh mới nhất
    latest_file += 1
else:
    latest_file = 0
saved = latest_file

# lặp qua các khung từ luồng tệp video
while True:
    # grab the frame from the file
    (grabbed, frame) = vs.read()
    
    # nếu khung hình không được nắm bắt, thì chúng ta đã xem đến cuối video
    if not grabbed:
        break
    
    # tăng số lượng khung đọc
    read += 1

    # kiểm tra xem chúng ta có nên xử lý khung này không
    # vì chúng ta muốn bỏ qua một số khung bổ trợ, chúng ta phải làm điều này
    if read % args['skip'] != 0:
        continue

    # lấy kích thước khung và tạo một đốm màu từ khung
    # về cơ bản, nó xử lý trước hình ảnh bằng phép trừ và chia tỷ lệ
    # (104.0, 177.0, 123.0) là trung bình của hình ảnh trong FaceNet
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                 (300,300), (104.0, 177.0, 123.0))
    
    # vượt qua blob thông qua NN và nhận được các phát hiện và dự đoán
    net.setInput(blob)
    detections = net.forward()
    
    # đảm bảo tìm thấy ít nhất một khuôn mặt
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
            face = frame[startY:endY, startX:endX]
            # ghi khung vào đĩa
            p = os.path.sep.join([args['output'], f'{saved}.png'])
            cv2.imwrite(p, face)
            saved += 1
            print(f'[INFO] saved {p} to disk')
            
# clean up
vs.release()
cv2.destroyAllWindows()