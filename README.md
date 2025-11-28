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
   source venv/bin/activate  # On Windows: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -> venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
<!-- pip freeze > requirements.txt -->

## Structure dataset

<pre>
    Structures's DATA REAL:
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

    Structures's DATA FAKE:
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