# Filter Mask

Trước tình hình diễn biến phức tạp của dịch covid19, việc đeo khẩu trang là biện pháp tối quan trọng trong việc phòng ngừa đại dịch. Nhưng nếu gọi người gọi điện trò chuyện online mà đeo khẩu trang thì hơi kì, chính vì thế filter mask là giải pháp để giúp mọi người thoải mái trò chuyện mà vẫn cùng cộng đồng chống dịch!

# How it work

Chúng ta sử dụng thư viện [Dlib](http://dlib.net/) (*shape_predictor_68_face_landmarks.dat*) để nhận diện các điểm landmarks trên khuôn mặt

![alt](https://raw.githubusercontent.com/KudoKhang/Filter-Face-Mask/main/figure_68_markup.jpg)

Dựa vào các điểm landmarks đó ta sẽ tính toán được vùng miệng sau đó dùng các hàm trong OpenCV để thực hiện ghép khẩu trang vào vùng miệng vừa tính toán ở trên

# Installations

Để khởi đầu tiên các bạn clone project này về máy

```
git clone https://github.com/KudoKhang/Filter-Face-Mask
```

Sau đó cài đặt một số thư viện cần thiết

```
pip install cv2
pip install numpy
pip install dlib
```



# Usage

Để khởi chạy các bạn mở terminal và: 

```
cd Filter-Face-Mask
python addMask.py
```



# Extend

- Dựa vào các điểm landmarks chúng ta có thể làm các filter thú vị như mũi heo, tai thỏ, lưỡi chó... như filter của facebook hay zalo.

- Các bạn có thể thử với

  ```
  addMosePig.py
  blurFaces.py (Làm mờ khuôn mặt)
  ```

  

### Note

- Với 68 landmarks của **Dlib** thì độ nhận diện chưa cao (như chúng ta nghiêng mặt chút thì không nhận diện được)

- Các bạn có thể thử với Mediapipe của google (với 468 landmarks, độ chính xác rất cao)
