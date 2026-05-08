# Medical AI Applied Projects

Deep learning applied to medical imaging and biometric analysis.

## 📓 Projects

| Notebook | Task | Dataset |
|---|---|---|
| `CNN_BrainTumorMRIScan.ipynb` | Brain tumor detection | Kaggle MRI (7,200 images, 4 classes: Glioma, Meningioma, No Tumor, Pituitary) |
| `Chest_X-Ray.ipynb` | Pneumonia detection | Chest radiograph dataset |
| `Skin_Cancer.ipynb` | Skin lesion classification | Dermoscopy images with augmentation |
| `EmotionDetectionWithFace.ipynb` | Facial emotion recognition | Face emotion categories |
| `UTKFace.ipynb` | Age estimation + gender classification | UTKFace dataset |

## 🧠 Brain Tumor MRI — 6 Architecture Comparison
All architectures trained on the same 5,600-image dataset with `ImageDataGenerator` (150×150, augmentation):

| Architecture | Type |
|---|---|
| Custom CNN | Conv2D(32→64→128) + Dense(512) + Dropout(0.5) |
| LeNet | 5×5 kernels, 2 conv-pool blocks |
| AlexNet | 11×11 stride-4 input, 5 conv layers, 3 FC layers |
| VGG16 | 13 conv layers, 3×3 kernels, 2×Dense(4096) |
| VGG19 | Pre-built from Keras Applications, frozen base |
| GoogLeNet | Inception-style, 25+ conv layers |

Predictions tested on real MRI images (`Glioma.jpg`, `NoTumor.jpg`) using OpenCV.
GPU required — runs on Colab T4.

## 🛠 Tech Stack
`Python` · `TensorFlow` · `Keras` · `OpenCV` · `kagglehub` · `NumPy` · `Pandas`

## 👤 Author
**Madhava Narra** — [@MadhavaCode](https://github.com/MadhavaCode) · Built May 2025 – May 2026
