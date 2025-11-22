# attend3d_model_ai

## Introduction

attend3d_model_ai is a project utilizing Artificial Intelligence (AI) and 3D modeling for [your purpose/industry, e.g., facial attendance, image recognition, data analysis, etc.].  
This project is developed by [Your team/individual name] with the goal of [general purpose].

## Features

- [Build/train] AI models for 3D data processing
- [Main feature 2, e.g., Face recognition, Image classification, etc.]
- [Integration with APIs/applications/...]
- [Scalability/extendability/... if any]

## Technologies Used

- Python 3.10 (Suggestion)
- [Deep Learning frameworks: TensorFlow, PyTorch, etc.]
- [3D processing libraries: Open3D, PCL, etc.]
- [Other supporting technologies/tools]

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nguyen-Phong-211/attend3d_model_ai.git
   cd attend3d_model_ai
   ```

2. Set up a virtual environment (recommended: virtualenv/anaconda):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- How to train/test the model:
   ```bash
   python train.py
   python test.py
   ```
- Instructions for using other scripts (if any).

## The commit history has been rewritten, so other machines will need to run:

```bash
git fetch --all
git reset --hard origin/main
```

## Contribution

Contributions are welcome! Please open a pull request or create an issue to discuss further.

## License

[MIT](LICENSE)  
Please see the LICENSE file for more details.

## Contact

- Author: zephyrnguyen.vn@gmail.com
- Github: [Nguyen-Phong-211](https://github.com/Nguyen-Phong-211)

<!-- 
Tôi có dự án nhận diện khuôn mặt 3D, bạn xem thử các file tôi đã làm đúng chưa, cần bổ sung gì không. Yêu cầu của tôi là model nhận diện chính xác, hoạt động tốt, nhận diện 3D, phân biệt thật giả, hoạt động tốt như FACE ID của IPHONE và các ứng dụng ngân hàng tại Việt Nam

Để tôi mô tả tôi có các tập dataset 300W_LP rồi sử dụng DECA để render ra ảnh thật, còn tập dataset còn lại thì vẫn sử dụng sử dụng DECA render ra ảnh giả.

Đây là đường dẫn dẫn đến tập dataset real để huấn luyện model nhận diện khuôn mặt thật

/Volumes/WD\ 500GB\ EL/DATA_ROOT/REAL 

Dưới dây là 2 đường dẫn dẫn đến 2 tập dataset fake để huấn luyện model nhận diện khuôn mặt giả

/Volumes/WD\ 500GB\ EL/DATA_ROOT/render_3d 
/Volumes/WD\ 500GB\ EL/DATA_ROOT/FAKE_RENDER 

Để tôi mô tả thêm, khi qua DECA chuyển đổi sang ảnh 3D thì sẽ phát sinh như sau

Ví dụ: Tôi có file frame_000001.jpg thì khi DECA chuyển ảnh thành 2 file và 1 thư mục sau

2 file: frame_000001_vis_original_size.jpg, frame_000001_vis.jpg
1 thư mục gồm có 5 file: frame_000001_depth.jpg, frame_000001_detail.obj, frame_000001_normals.png, frame_000001.mtl, frame_000001.obj, frame_000001.png
 -->

<pre>
    Cấu trúc DATA REAL:
    DATA_ROOT/REAL/
      AFW/
        AFW_134212_1_0/
            AFW_134212_1_0_depth.jpg, 
            AFW_134212_1_0_detail.obj, 
            AFW_134212_1_0_normals.png, 
            AFW_134212_1_0.mtl, 
            AFW_134212_1_0.obj, 
            AFW_134212_1_0.png
        AFW_134212_1_0_vis_original_size.jpg, 
        AFW_134212_1_0_vis.jpg

    Cấu trúc DATA FAKE:
    DATA_ROOT/FAKE_RENDER/
        easy_1_1110/
            easy_1_1110_depth.jpg, 
            easy_1_1110_detail.obj, 
            easy_1_1110_normals.png, 
            easy_1_1110.mtl, 
            easy_1_1110.obj, 
            easy_1_1110.png
        easy_1_1110_vis_original_size.jpg, 
        easy_1_1110_vis.jpg

    DATA_ROOT/render_3d/
        original_000/
            frame_000001_depth.jpg, 
            frame_000001_detail.obj, 
            frame_000001_normals.png, 
            frame_000001.mtl, 
            frame_000001.obj, 
            frame_000001.png
        frame_000001_vis_original_size.jpg, 
        frame_000001_vis.jpg
</pre>