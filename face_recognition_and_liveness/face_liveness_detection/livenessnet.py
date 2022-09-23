import tensorflow as tf

class LivenessNet:
    
    @staticmethod
    def build(width, height, depth, classes):
        # khởi tạo mô hình cùng với hình dạng đầu vào để được
        # 'kênh cuối cùng' và chính thứ nguyên kênh
        INPUT_SHAPE = (height, width, depth)
        chanDim = -1 # sử dụng để chuẩn hóa hàng loạt dọc theo trục

         # nếu chúng tôi đang sử dụng "các kênh trước tiên", hãy cập nhật hình dạng đầu vào
         # và thứ nguyên kênh
         # lưu ý rằng: thông thường, theo mặc định, đó là "kênh cuối cùng"
        if tf.keras.backend.image_data_format() == 'channels_first':
            INPUT_SHAPE = (depth, height, width)
            chanDim = 1

        # CNN của chúng tôi thể hiện những phẩm chất VGGNet-esque. Nó rất nông với chỉ một số bộ lọc đã học.
        # Tốt nhất, chúng tôi sẽ không cần một mạng lưới sâu rộng để phân biệt giữa khuôn mặt thật và khuôn mặt giả mạo.
        model = tf.keras.Sequential([
                # first set CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu' ,input_shape=INPUT_SHAPE),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # second set CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(axis=chanDim),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # FullyConnected => BatchNorm => Dropout
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                # đầu ra
                tf.keras.layers.Dense(classes, activation='softmax')
            ])
        
        return model