```markdown
# Nail Segmentation with U-Net using PyTorch

## 📌 Mô tả

Dự án này triển khai mô hình **U-Net** sử dụng thư viện **PyTorch** nhằm **phân đoạn vùng móng tay (nail segmentation)** trên ảnh đầu vào. Mục tiêu là tạo ra mặt nạ phân đoạn chính xác, hỗ trợ các ứng dụng như làm đẹp, chăm sóc móng, hoặc tiền xử lý trong các hệ thống nhận diện tay.

---

## 🧠 Kiến thức sử dụng

- Deep Learning
- Semantic Segmentation
- Kiến trúc U-Net
- PyTorch
- Xử lý ảnh với OpenCV
- Chia tập huấn luyện / kiểm tra

---

## 📁 Cấu trúc thư mục

```

nail-segmentation-unet/
│
├── data/                   # Thư mục chứa dữ liệu đầu vào
│   ├── train/
│   ├── val/
│   └── test/
│
├── model/                  # Lưu mô hình đã huấn luyện
│
├── outputs/                # Lưu kết quả sau phân đoạn
│
├── unet.py                 # Định nghĩa mô hình U-Net
├── train.py                # File huấn luyện mô hình
├── predict.py              # Dự đoán ảnh mới bằng mô hình đã huấn luyện
├── utils.py                # Các hàm hỗ trợ xử lý ảnh, hiển thị kết quả,...
├── requirements.txt        # Danh sách thư viện cần cài
└── README.md               # Tài liệu mô tả dự án

````

---


## 🚀 Hướng dẫn chạy

### 1. Cài thư viện cần thiết

Tạo virtual environment và cài đặt:
```bash
pip install -r requirements.txt
````

### 2. Chuẩn bị dữ liệu

Tải ảnh và nhãn (mask) vào thư mục `data/train`, `data/val` theo định dạng:

* Ảnh: `.jpg` hoặc `.png`
* Mặt nạ: ảnh trắng đen, cùng kích thước

<img width="831" height="329" alt="image" src="https://github.com/user-attachments/assets/08c93efe-e57c-4d48-a9ae-b0421d7ad955" />


### 3. Huấn luyện mô hình

```bash
python train.py
```

### 4. Dự đoán trên ảnh mới


<img width="993" height="339" alt="image" src="https://github.com/user-attachments/assets/5da84c46-8b7c-438c-9c80-f75df77285eb" />



## 🛠️ Thư viện sử dụng

* `torch`
* `torchvision`
* `numpy`
* `opencv-python`
* `matplotlib`
* `tqdm`

---

## 📈 Kết quả huấn luyện

* Độ chính xác (IoU): \~**XX%**
* Loss giảm đều qua các epoch

<img width="1304" height="620" alt="image" src="https://github.com/user-attachments/assets/1815bbeb-d33e-4242-9936-cca249053f8c" />


---

## 📚 Tài liệu tham khảo

* Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
* Segmenting Nails with Deep Learning (blogs)

---

## 📩 Liên hệ

> Nếu bạn thấy dự án hữu ích, hãy star ⭐ repo này nhé!
> Mọi góp ý hoặc thắc mắc(hiepbt17@gmail.com).

---

