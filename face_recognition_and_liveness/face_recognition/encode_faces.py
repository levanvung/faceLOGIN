from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# xây dựng trình phân tích cú pháp đối số và phân tích cú pháp đối số
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataset', required=True,
                    help='Path to input directory of faces + images')
parser.add_argument('-e', '--encoding', required=True,
                    help='Path to save encoded images pickle')
parser.add_argument('-d', '--detection-method', type=str, default='cnn',
                    help="Face detection model to use: 'hog' or 'cnn'")
args = vars(parser.parse_args())
# lấy các đường dẫn đến hình ảnh đầu vào trong tập dữ liệu của chúng tôi
print('[INFO] quantifying faces...')
imagePaths = list(paths.list_images(args['dataset']))

# khởi tạo danh sách các mã hóa đã biết và các tên đã biết
knownEncodings = list()
knownNames = list()

# lặp qua đường dẫn đến từng hình ảnh
for (i, imagePath) in enumerate(imagePaths):
    # trích xuất tên từ đường dẫn
    # ví dụ: đường dẫn: dataset / name / image1.jpg
    # nếu chúng ta sử dụng os.path.sep để tách nó
    # và chọn thứ hai từ chỉ mục cuối cùng
    # chúng ta sẽ nhận được 'tên', trong trường hợp này, là một nhãn
    print(f'[INFO] processing images {i+1}/{len(imagePaths)}')
    name = imagePath.split(os.path.sep)[-2]

    # tải hình ảnh và chuyển đổi rom BGR (OpenCV mặc định)
    # sang RGB (mặc định dlib)
    image = cv2.imread(imagePath)
    # giảm kích thước xuống một nửa để xử lý nhanh hơn
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # phát hiện khuôn mặt trong mỗi khung hình
    # và trả về (x, y) -phần phụ của hộp giới hạn
    boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
    
    # tính toán nhúng khuôn mặt
    encodings = face_recognition.face_encodings(rgb, boxes)

    # lặp qua các mã hóa
    # lý do chúng tôi phải lặp lại mã hóa ngay cả khi đó là một hình ảnh duy nhất
    # là đôi khi khuôn mặt của một người có thể xuất hiện ở nhiều vị trí trong ảnh
    # ví dụ: người đó đang soi gương
    for encoding in encodings:
        # add each encoding and name to the list
        knownEncodings.append(encoding)
        knownNames.append(name)
        
# thêm từng bảng mã và tên vào danh sách
print('[INFO] saving encodings...')
data = {'encodings': knownEncodings, 'names': knownNames}
with open(args['encoding'], 'wb') as file:
    file.write(pickle.dumps(data))
    