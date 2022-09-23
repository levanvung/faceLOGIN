import os
# uncomment this line if you want to run your tensorflow model on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from imutils.video import VideoStream
import face_recognition
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2

# if you want to run this file from the shell,
# uncomment these lines below and delete the function header and return

# # construct the argument parser and parse the arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model', type=str, required=True,
#                     help='Path to trained model')
# parser.add_argument('-l', '--le', type=str, required=True,
#                     help='Path to Label Encoder')
# parser.add_argument('-d', '--detector', type=str, required=True,
#                     help='Path to OpenCV\'s deep learning face detector')
# parser.add_argument('-c', '--confidence', type=float, default=0.5,
#                     help='minimum probability to filter out weak detections')
# parser.add_argument('-e', '--encodings', required=True,
#                     help='Path to saved face encodings')
# args = vars(parser.parse_args())

def recognition_liveness(model_path, le_path, detector_folder, encodings, confidence=0.5):
    args = {'model':model_path, 'le':le_path, 'detector':detector_folder, 
            'encodings':encodings, 'confidence':confidence}

    # tải các khuôn mặt và tên được mã hóa
    print('[INFO] loading encodings...')
    with open(args['encodings'], 'rb') as file:
        encoded_data = pickle.loads(file.read())
    # tải máy dò khuôn mặt được tuần tự hóa của chúng tôi từ đĩa
    print('[INFO] loading face detector...')
    proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
    model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
    detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    # tải mô hình phát hiện độ sống và bộ mã hóa nhãn từ đĩa
    liveness_model = tf.keras.models.load_model(args['model'])
    le = pickle.loads(open(args['le'], 'rb').read())

    # khởi chạy luồng video và cho phép máy ảnh khởi động
    print('[INFO] starting video stream...')
    vs = VideoStream(src=0).start()
    time.sleep(2)# đợi máy ảnh khởi động
     # đếm chuỗi người đó xuất hiện
     # điều này chỉ để chắc chắn về người đó và để hiển thị cách hoạt động của mô hình
     # bạn có thể xóa cái này nếu bạn muốn
    sequence_count = 0 
    
    # lặp lại các khung từ luồng video
    while True:
        # lấy khung hình từ luồng video theo chuỗi
        # và thay đổi kích thước để có chiều rộng tối đa là 600 pixel
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        cv2.putText(frame, "Press 'q' to quit", (20,35), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 2)
        # lấy các kích thước khung và chuyển đổi nó thành một đốm màu
        # blob được sử dụng để xử lý trước hình ảnh để dễ đọc cho NN
        # về cơ bản, nó có nghĩa là trừ và chia tỷ lệ
        # (104.0, 177.0, 123.0) là trung bình của hình ảnh trong FaceNet
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # vượt qua các đốm màu qua mạng
        # và nhận được các phát hiện và dự đoán
        detector_net.setInput(blob)
        detections = detector_net.forward()

        # lặp lại các phát hiện
        for i in range(0, detections.shape[2]):
            #trích xuất độ tin cậy (tức là xác suất) được kết hợp với dự đoán
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections
            if confidence > args['confidence']:
                # tính tọa độ (x, y) của hộp giới hạn
                # cho khuôn mặt và trích xuất ROI của khuôn mặt
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # mở rộng hộp giới hạn một chút
                # (từ thử nghiệm, mô hình hoạt động tốt hơn theo cách này)
                # và đảm bảo rằng hộp giới hạn không nằm ngoài khung
                startX = max(0, startX-20)
                startY = max(0, startY-20)
                endX = min(w, endX+20)
                endY = min(h, endY+20)

                # trích xuất ROI của khuôn mặt và sau đó xử lý trước
                # theo cách tương tự như dữ liệu đào tạo của chúng tôi
                face = frame[startY:endY, startX:endX] # để phát hiện độ sống
                 # mở rộng hộp giới hạn để mô hình có thể xác định lại dễ dàng hơn
                face_to_recog = face # để công nhận
                 # một số lỗi xảy ra ở đây nếu khuôn mặt của tôi nằm ngoài khung hình và quay trở lại trong khung hình
                try:
                    face = cv2.resize(face, (32,32)) # mô hình liveness của chúng tôi mong đợi đầu vào 32x32
                except:
                    break

                # nhận dạng khuôn mặt
                rgb = cv2.cvtColor(face_to_recog, cv2.COLOR_BGR2RGB)
                # rgb = cv2.cvtColor (khuôn mặt, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                # khởi tạo tên mặc định nếu nó không tìm thấy khuôn mặt cho các khuôn mặt được phát hiện
                name = 'Unknown'
                # vòng lặp qua các mặt được mã hóa (thậm chí nó chỉ có 1 mặt trong một hộp giới hạn)
                # đây chỉ là quy ước cho các tác phẩm khác với loại mô hình này
                for encoding in encodings:
                    matches = face_recognition.compare_faces(encoded_data['encodings'], encoding)

                    # kiểm tra xem chúng tôi có tìm thấy khuôn mặt phù hợp không
                    if True in matches:
                        # tìm chỉ mục của tất cả các mặt đã khớp sau đó khởi tạo một chính tả
                        # để đếm tổng số lần mỗi khuôn mặt được khớp
                        matchedIdxs = [i for i, b in enumerate(matches) if b]
                        counts = {}
                        
                        # vòng lặp qua các chỉ mục phù hợp và đếm
                        for i in matchedIdxs:
                            name = encoded_data['names'][i]
                            counts[name] = counts.get(name, 0) + 1
                            
                        # lấy tên có số lượng nhiều nhất
                        name = max(counts, key=counts.get)
                            
                face = face.astype('float') / 255.0 
                face = tf.keras.preprocessing.image.img_to_array(face)
                # Mô hình tf yêu cầu hàng loạt dữ liệu để cấp vào
                #                  # vì vậy nếu chúng tôi chỉ cần một hình ảnh mỗi lần, chúng tôi phải thêm một kích thước nữa
                #                  # trong trường hợp này giống với [khuôn mặt]
                face = np.expand_dims(face, axis=0)

                # vượt qua ROI của khuôn mặt thông qua mô hình phát hiện trực tiếp được đào tạo
                # để xác định xem khuôn mặt là 'thật' hay 'giả'
                # dự đoán giá trị trả về 2 cho mỗi ví dụ (vì trong mô hình chúng ta có 2 lớp đầu ra)
                # giá trị đầu tiên lưu trữ xác suất là thật, giá trị thứ hai lưu trữ xác suất là giả
                # so argmax sẽ chọn người có prob cao nhất
                # chúng tôi chỉ quan tâm đến đầu ra đầu tiên (vì chúng tôi chỉ có 1 đầu vào)
                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_name = le.classes_[j] # lấy nhãn của lớp được dự đoán

                 # vẽ nhãn và hộp giới hạn trên khung
                label = f'{label_name}: {preds[j]:.4f}'
                if name == 'Unknown' or label_name == 'fake':
                    sequence_count = 0
                else:
                    sequence_count += 1
                print(f'[INFO] {name}, {label_name}, seq: {sequence_count}')
                
                if label_name == 'fake':
                    cv2.putText(frame, "Don't try to Spoof !", (startX, endY + 25), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
                
                cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)

        # hiển thị danh tiếng đầu ra và chờ một lần nhấn phím
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # nếu 'q' được nhấn, dừng vòng lặp
        # nếu người đó xuất hiện 10 khung hình liên tiếp, hãy dừng vòng lặp lại
        # bạn có thể thay đổi điều này nếu GPU của bạn chạy nhanh hơn
        if key == ord('q') or sequence_count==10:
            break
        
    # cleanup
    vs.stop()
    cv2.destroyAllWindows()
    # have some times for camera and CUDA to close normally
    # (it can f*ck up GPU sometimes if you don't have high performance GPU like me LOL)
    time.sleep(2)
    return name, label_name
        
if __name__ == '__main__':
    name, label_name = recognition_liveness('liveness.model', 'label_encoder.pickle', 
                                            'face_detector', '../face_recognition/encoded_faces.pickle', confidence=0.5)
    print(name, label_name)
        