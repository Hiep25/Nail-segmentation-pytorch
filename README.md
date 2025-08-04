````markdown
  💅 Nail Segmentation with U-Net using PyTorch

 📌 Mô tả

Dự án này triển khai mô hình **U-Net** sử dụng thư viện **PyTorch** nhằm **phân đoạn vùng móng tay (nail segmentation)** trên ảnh RGB đầu vào. Mục tiêu là tạo ra mặt nạ phân đoạn chính xác, phục vụ các ứng dụng như: làm đẹp, chăm sóc móng tự động, trang điểm AR, hoặc tiền xử lý trong các hệ thống thị giác máy tính.

---

 🧠 Kiến thức và kỹ thuật sử dụng

- Học sâu (Deep Learning)
- Semantic Segmentation
- Kiến trúc U-Net
- PyTorch
- Xử lý ảnh với OpenCV
- Chia dữ liệu train / val / test

---

 📁 Cấu trúc thư mục

```bash
nail-segmentation-unet/
├── data/                   # Thư mục dữ liệu đầu vào (ảnh + mask)
│   ├── train/
│   ├── val/
│   └── test/
├── model/                  # Lưu trọng số mô hình đã huấn luyện
├── outputs/                # Kết quả phân đoạn và hình ảnh trực quan
├── unet.py                 # Định nghĩa kiến trúc mô hình U-Net
├── train.py                # Huấn luyện mô hình
├── predict.py              # Dự đoán ảnh mới
├── utils.py                # Hàm hỗ trợ: xử lý ảnh, hiển thị, v.v.
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md               # Mô tả dự án (file này)
````

---

 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

* Đặt ảnh và mask tương ứng trong `data/train`, `data/val`, `data/test`
* Ảnh: định dạng `.jpg` hoặc `.png`
* Mặt nạ (mask): ảnh trắng đen (0 và 255), cùng kích thước với ảnh gốc

📌 **Ví dụ:**

![Input & Mask Example](https://github.com/user-attachments/assets/08c93efe-e57c-4d48-a9ae-b0421d7ad955)

---

### 3. Huấn luyện mô hình

```bash
python train.py
```

* Quá trình huấn luyện hiển thị loss và lưu mô hình vào thư mục `model/`
* Có thể tùy chỉnh tham số trong file `train.py` hoặc `config.py` (nếu có)

---

### 4. Dự đoán trên ảnh mới

```bash
python predict.py --image path/to/image.jpg
```

📸 Ví dụ kết quả phân đoạn:

![Prediction Output](https://github.com/user-attachments/assets/5da84c46-8b7c-438c-9c80-f75df77285eb)

---

 🛠️ Thư viện sử dụng

* `torch`
* `torchvision`
* `numpy`
* `opencv-python`
* `matplotlib`
* `tqdm`

---

## 📈 Kết quả huấn luyện

- 🎯 **Dice Score (trung bình trên tập kiểm tra): ~92.39%**
- 📉 **Loss giảm đều và ổn định qua các epoch**
- 📊 Một số batch đạt Dice Score cao tới **96.00%**

## 📊 Minh họa quá trình huấn luyện:

![Training Curve](https://github.com/user-attachments/assets/1815bbeb-d33e-4242-9936-cca249053f8c)

---

## 📚 Tài liệu tham khảo

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* Bài viết: Segmenting Nails with Deep Learning (blogs, medium, etc.)

---

## 📩 Liên hệ

> Nếu bạn thấy dự án hữu ích, hãy ⭐ Star repo để ủng hộ!
> Mọi góp ý hoặc thắc mắc xin gửi về: **[hiepbt17@gmail.com](mailto:hiepbt17@gmail.com)**

---
