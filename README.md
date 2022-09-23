vapp.py:Đây là ứng dụng Flask chính. Để sử dụng nó, chỉ cần chạy tệp này từ thiết bị đầu cuối / IDE ưa thích của bạn và cổng chạy cho ứng dụng web của bạn sẽ được hiển thị. Nếu bạn muốn triển khai ứng dụng này, đừng quên thay đổi biến app.secret_key
database.sqlite:Đây là ví dụ tối thiểu về cơ sở dữ liệu để lưu trữ dữ liệu của người dùng và nó được sử dụng để truy xuất và xác minh trong khi người dùng đang đăng nhập.
Trong cặpface_recognition_and_liveness/ face_recognition
encode_faces.py:Phát hiện khuôn mặt từ hình ảnh, mã hóa và lưu phiên bản và tên được mã hóa (nhãn) vào các tệp dưa chua. Tên/nhãn coms từ tên của thư mục của những hình ảnh đầu vào đó.
Đối số dòng lệnh:
--dataset (or -i) Đường dẫn nhập thư mục của hình ảnh
--encoding (or -e) Đường dẫn/Thư mục để lưu tệp dưa chua hình ảnh được mã hóa
--detection-method (or -d) Mô hình phát hiện khuôn mặt để sử dụng: 'hog' hoặc 'cnn' (mặc định là 'cnn')
Ví dụ: python encode_faces.py -i dataset -e encoded_faces.pickle -d cnn
recognize_faces.py:Nhận dạng khuôn mặt theo thời gian thực trên webcam với các hộp giới hạn và tên hiển thị trên đầu trang.
Đối số dòng lệnh:
--encoding (or -e) Đường dẫn đến mã hóa khuôn mặt đã lưu
--detection-method (or -d) Mô hình phát hiện khuôn mặt để sử dụng: 'hog' hoặc 'cnn' (mặc định là 'cnn')
Ví dụ: python recognize_faces.py -e encoded_faces.pickle -d cnn
thư mục dataset: Cặp ví dụ và hình ảnh để lưu trữ bộ dữ liệu được sử dụng để mã hóa khuôn mặt. Có cặp con trong thư mục này và mỗi cặp con phải được đặt tên theo chủ sở hữu hình ảnh trong cặp con đó vì tên thư mục con này sẽ được sử dụng làm nhãn trong quá trình mã hóa khuôn mặt.
Trong cặpface_recognition_and_liveness/ face_liveness_detection
collect_dataset.py:Thu thập khuôn mặt trong mỗi khung hình từ bộ dữ liệu video (thật / giả) bằng mô hình máy dò khuôn mặt (SSD resnet-10 trong
trường hợp này) và lưu vào thư mục (chúng tôi đã cung cấp ví dụ video trong thư mục video, vì vậy bạn có thể thu thập bộ dữ liệu video chính xác để đào tạo mô hình) Đối số dòng lệnh:
--input (or -i) Đường dẫn đến nhập video
--output (or -o) Path/Directory to output directory of face images cắt xén
--detector (or -d) Đường đến máy dò khuôn mặt học sâu của OpenCV
--confidence (or -c) Độ tin cậy của mô hình máy dò khuôn mặt (mặc định là 0,5 | 50%)
--skip (or -s) Số khung hình cần bỏ qua trước khi áp dụng phát hiện khuôn mặt và cắt xén (dafault là 16). Ý tưởng chính cho điều này là các khung hậu quả thường đưa ra cùng một khuôn mặt cho bộ dữ liệu, vì vậy nó có thể dễ dàng gây ra sự phù hợp quá mức và không phải là dữ liệu hữu ích cho đào tạo.
Ví dụ: ví dụ về bộ dữ liệu video giả mạo -> | ví dụ về bộ dữ liệu video thực -> python collect_dataset.py -i videos/fake_1.mp4 -o dataset/fake -d face_detector -c 0.5 -s 15python collect_dataset.py -i videos/real_1.mp4 -o dataset/real -d face_detector -c 0.5 -s 15
face_from_image.py:Thu thập khuôn mặt trong mỗi khung hình từ bộ dữ liệu hình ảnh (thật / giả) bằng cách sử dụng mô hình máy dò khuôn mặt (RESnet-10 SSD trong trường hợp này) và lưu vào thư mục (chúng tôi đã cung cấp ví dụ video trong thư mục video, vì vậy bạn có thể thu thập bộ dữ liệu video chính xác để đào tạo mô hình)
Đối số dòng lệnh:
--input (or -i) Đường dẫn để nhập ảnh (Một hình ảnh duy nhất | Vì chúng tôi chủ yếu thu thập bộ dữ liệu từ video, chúng tôi chỉ sử dụng mã này để thu thập khuôn mặt từ những hình ảnh in rắn đó (hình ảnh từ giấy / thẻ) và chúng tôi không có nhiều trong số chúng. Vì vậy, chúng tôi làm mã chỉ để thu thập khuôn mặt từ 1 hình ảnh. Hãy thoải mái điều chỉnh mã nếu bạn muốn làm cho nó có thể thu thập khuôn mặt từ tất cả các hình ảnh trong thư mục / thư mục)
--output (or -o) Path/Directory to output directory of face images cắt xén
--detector (or -d) Đường đến máy dò khuôn mặt học sâu của OpenCV
--confidence (or -c) Độ tin cậy của mô hình máy dò khuôn mặt (mặc định là 0,5 | 50%)
Ví dụ:ví dụ cho bộ dữ liệu hình ảnh giả mạo -> | ví dụ về bộ dữ liệu hình ảnh thực -> python face_from_image.py -i images/fakes/2.jpg -o dataset/fake -d face_detector -c 0.5python face_from_image.py -i images/reals/1.jpg -o dataset/real -d face_detector -c 0.5
livenessnet.py:Kiến trúc mô hình cho mô hình phát hiện sự sống động của chúng tôi và xây dựng chức năng để xây dựng mạng thần kinh (không có đối số dòng lệnh cho tệp này (không cần phải làm điều đó)). Lớp LivenessNet sẽ được gọi từ tệp để xây dựng một mô hình và chạy quá trình đào tạo từ tệp train_model.py.train_model.py
train_model.py:Mã được sử dụng để đào tạo mô hình phát hiện sự sống động và đầu ra .model, label_encoder.dưa chua và các tệp hình ảnh.png cốt truyện.
Đối số dòng lệnh:
--dataset (or -d) Đường dẫn đến Bộ dữ liệu nhập
--model (or -m) Con đường đến mô hình được đào tạo đầu ra
--le (or -l) Đường dẫn đến Bộ mã hóa nhãn đầu ra
--plot (or -p) Đường dẫn đến lỗ
đầu ra / cốt truyện chính xác Ví dụ: python train_model.py -d dataset -m liveness.model -l label_encoder.pickle -p plot.png
liveness_app.py:Chạy phát hiện khuôn mặt, vẽ hộp giới hạn và
chạy mô hình phát hiện sự sống theo thời gian thực trên đối số dòng lệnh webcam:
--model (or -m) Con đường đến mô hình được đào tạo
--le (or -l) Đường dẫn đến Bộ mã hóa Nhãn
--detector (or -d) Đường đến máy dò khuôn mặt học sâu của OpenCV
--confidence (or -c) Độ tin cậy của mô hình máy dò khuôn mặt (mặc định là 0,5 | 50%)
Ví dụ: python liveness_app.py -m liveness.model -l label_encoder.pickle -d face_detector -c 0.5
face_recognition_liveness_app.py:Đây là tệp cốt lõi kết hợp cả nhận dạng khuôn mặt và phát hiện sự sống động với nhau và chạy chúng đồng thời. Phiên bản hiện tại của tệp này được tái cấu trúc để sử dụng trong tệp chính, vì vậy tệp này không hỗ trợ đối số dòng lệnh. Tuy nhiên,chúng tôi đã cung cấp mã cho dòng lệnh bên trong chính mã nguồn và nhận xét những dòng đó. Nếu bạn thực sự muốn chạy hoàn toàn từ dòng lệnh, hãy tháo các dòng đó và nhận xét cấu trúc hàm (tiêu đề, trả về và nếu __name__ == '__main__') và đó là nó. Trong trường hợp bạn muốn chạy từ dòng lệnh, chúng tôi sẽ cung cấp đối số dòng lệnh và ví dụ ở đây.
Ví dụ nếu bạn không
sửa đổi mã : Đối số dòng lệnh: app.pypython face_recognition_liveness_app.py
--model (or -m) Con đường đến mô hình được đào tạo
--le (or -l) Đường dẫn đến Bộ mã hóa Nhãn
--detector (or -d) Đường đến máy dò khuôn mặt học sâu của OpenCV
--confidence (or -c) Độ tin cậy của mô hình máy dò khuôn mặt (mặc định là 0,5 | 50%)
--encodings (or -e) Đường dẫn đến mã hóa
khuôn mặt đã lưu Ví dụ nếu bạn sửa đổi mã và sử dụng đối số dòng lệnh: python face_recognition_liveness_app.py -m liveness.model -l label_encoder.pickle -d face_detector -c 0.5 -e ../face_recognition/encoded_faces.pickle
thư mục bộ dữ liệu: Thư mục và hình ảnh ví dụ để đào tạo mô hình phát hiện tính sống. (Những hình ảnh này là đầu ra của collect_dataset.py)
thư mục face_detector: Thư mục chứa các tệp mô hình caffe bao gồm .prototxt và .caffemodel để sử dụng với OpenCV và thực hiện phát hiện khuôn mặt
cặp hình ảnh: Cặp ví dụ và hình ảnh để nhập vào face_from_image.py
cặp video: Cặp ví dụ và video để nhập vào collect_dataset.py



===========================================================================================================================
Tạo 1 thư mục cho 1 người và đặt tên theo tên của người đó trong face_recognition / dataset (bạn có thể xem thư mục này trong repo này chẳng hạn)
Thu thập các hình ảnh hiển thị đầy đủ khuôn mặt (1 khuôn mặt trên 1 hình ảnh trên 1 người). Vì chúng tôi đang sử dụng kỹ thuật học chụp 1 lần, nên chỉ thu thập tối đa 10 hình ảnh cho mỗi người là đủ.
Chạy encode_faces.py giống như ví dụ ở trên trong phần giải thích tệp
Bây giờ bạn sẽ nhận được tệp khuôn mặt được mã hóa kết thúc bằng .pickle trong đường dẫn bạn chỉ định (nếu bạn làm theo mã ở trên, bạn sẽ thấy nó trong cùng một thư mục với tệp này)
Chạy Recog_faces.py giống như ví dụ ở trên trong phần giải thích tệp và xem liệu nó có hoạt động tốt hay không.
Thu thập video của chính bạn / người khác trong nhiều điều kiện ánh sáng (cách dễ nhất để làm điều này là quay phim chính bạn / người khác đang đi quanh nhà của bạn / những người khác) và lưu vào thư mục face_liveness_dection / videos. Độ dài của video tùy thuộc vào bạn. Bạn không cần phải đặt tên cho nó bằng từ 'thật' hoặc 'giả'. Đó chỉ là quy ước mà chúng tôi thấy hữu ích khi gọi từ các mã khác. Hãy xem thư mục đó, chúng tôi đã bỏ một số video ví dụ ở đó.
Sử dụng những video đã ghi đó và phát nó trên điện thoại của bạn. Sau đó, giữ điện thoại của bạn và hướng màn hình điện thoại (đang chạy các video đã quay đó) vào webcam và quay màn hình PC / laptop của bạn. Bằng cách làm này, bạn đang tạo tập dữ liệu về ai đó giả mạo người trong video / giả mạo là người trong video. Cố gắng đảm bảo video giả mạo mới này có cùng độ dài (hoặc gần) với video gốc vì chúng ta cần tránh tập dữ liệu không cân bằng. Hãy xem thư mục đó, chúng tôi đã bỏ một số video ví dụ ở đó.
Chạy collect_dataset.py giống như ví dụ ở trên trong phần giải thích tệp cho mọi video của bạn. Đảm bảo rằng bạn lưu kết quả đầu ra vào đúng thư mục (phải nằm trong thư mục tập dữ liệu và trong thư mục nhãn bên phải giả hay thật). Bây giờ bạn phải xem rất nhiều hình ảnh từ video của mình trong thư mục đầu ra.
(Tùy chọn, nhưng nên làm để cải thiện hiệu suất của mô hình) Chụp ảnh bạn / người khác từ giấy, ảnh, thẻ, v.v. và lưu vào face_liveness_detection / images / fakes. Hãy xem thư mục đó, chúng tôi đã bỏ một số video ví dụ ở đó.
Nếu bạn làm bước 9, vui lòng thực hiện bước này. Nếu không, bạn có thể bỏ qua bước này. Chụp thêm ảnh khuôn mặt của bạn / người khác với cùng lượng ảnh giả mà bạn đã chụp ở bước 8 và lưu vào face_liveness_detection / images / reals. Một lần nữa, bằng cách làm này, chúng ta có thể tránh được tập dữ liệu không cân bằng. Hãy xem thư mục đó, chúng tôi đã bỏ một số video ví dụ ở đó.
(Bỏ qua bước này nếu bạn chưa thực hiện bước 9) Chạy face_from_image.py giống như ví dụ ở trên trong phần giải thích tệp cho mọi hình ảnh của bạn trong thư mục hình ảnh. Đảm bảo rằng bạn lưu kết quả đầu ra vào đúng thư mục (phải nằm trong thư mục tập dữ liệu và trong thư mục nhãn bên phải giả hay thật). Lưu ý: Giống như chúng ta đã thảo luận trong phần giải thích tệp, bạn phải chạy mã này 1 hình ảnh tại một thời điểm. Nếu bạn có nhiều hình ảnh, vui lòng điều chỉnh mã. Vì vậy, bạn chỉ có thể chạy một lần cho mỗi hình ảnh của mình. (Nhưng hãy đảm bảo lưu kết quả đầu ra vào đúng thư mục)
Chạy train_model.py giống như ví dụ ở trên trong phần giải thích tệp. Bây giờ, chúng ta sẽ có .model, tệp bộ mã hóa nhãn kết thúc bằng .pickle và hình ảnh trong thư mục đầu ra mà bạn chỉ định. Nếu bạn làm theo mã chính xác ở trên trong phần giải thích tệp, bạn sẽ thấy liveness.model, label_encoder.pickle và plot.png trong thư mục chính xác này (như trong repo này).
Chạy liveness_app.py giống như ví dụ ở trên trong phần giải thích tệp và xem liệu nó có hoạt động tốt hay không. Nếu mô hình luôn phân loại sai, hãy quay lại và xem liệu bạn có lưu hình ảnh đầu ra (thật / giả) vào đúng thư mục hay không. Nếu bạn chắc chắn rằng bạn lưu mọi thứ vào đúng vị trí, hãy thu thập nhiều dữ liệu hơn hoặc dữ liệu chất lượng tốt hơn. Đây là quá trình lặp đi lặp lại phổ biến của mô hình đào tạo, đừng cảm thấy tồi tệ nếu bạn gặp vấn đề này.
Chạy face_recognition_liveness_app.py giống như ví dụ ở trên trong phần giải thích tệp và xem liệu nó có hoạt động tốt hay không. Cửa sổ sẽ hiển thị một lúc với hộp giới hạn có tên của bạn ở trên cùng và xác suất là thật hay giả. Nếu mã của bạn hoạt động bình thường, cửa sổ này sẽ tự động đóng lại sau khi mô hình có thể phát hiện ra bạn và bạn thực sự có 10 khung hệ quả. Bằng cách chỉ định 10 khung hệ quả này, nó có thể đảm bảo rằng người trên màn hình thực sự là người đang đăng nhập và có thật, chứ không chỉ do ngẫu nhiên mà mô hình phân loại sai cho một số khung. Nó cũng cho phép một số chỗ để mô hình phân loại sai. Bạn có thể điều chỉnh con số này trong biến 'số_trình_số_số' dòng 176 trong mã, nếu bạn muốn nó dài hơn hoặc ngắn hơn. Và ngay sau khi cửa sổ đóng lại, chúng ta sẽ thấy tên và nhãn (thật) của bạn trên dòng lệnh / thiết bị đầu cuối.
Trong app.py, trong thư mục chính của dự án này, đi xuống dòng 56, v.v. Bỏ ghi chú những dòng đó và thay đổi các tham số trong đối tượng Người dùng thành tên người dùng, mật khẩu và tên của bạn. Nếu bạn muốn thêm cột, hãy chuyển đến dòng 13 và thêm nhiều cột vào đó. LƯU Ý QUAN TRỌNG: Tên bạn lưu ở đây