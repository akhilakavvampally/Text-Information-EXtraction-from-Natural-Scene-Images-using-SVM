
# 🧠 Text Extraction & Image Classification Pipeline

This project implements a complete pipeline for image-based text region detection, feature extraction with HOG, classification using Support Vector Machines (SVM), and Optical Character Recognition (OCR) using Tesseract.

## 📌 Features

- 📷 Preprocess grayscale images using histogram equalization.
- 🔍 Feature extraction with Histogram of Oriented Gradients (HOG).
- 🧪 Dimensionality reduction via PCA.
- 🧠 SVM-based classification with GridSearchCV for hyperparameter tuning.
- 🧾 Text detection using MSER and OCR using Tesseract.
- 📊 Evaluation with accuracy, classification report, and confusion matrix.
- 🖼️ Image visualization with bounding boxes and intermediate stages.

---

## 🚀 Getting Started

### ✅ Requirements

Install the following before running the notebook:

```bash
sudo apt install tesseract-ocr -y
pip install pytesseract numpy opencv-python matplotlib seaborn scikit-learn
```

Ensure Tesseract is correctly linked in your script:

```python
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

---

## 🗂️ Dataset

Your dataset directory should be structured as:

```
/minor dataset (ICDAR)/
  ├── class1/
  │    ├── image1.jpg
  │    └── image2.jpg
  └── class2/
       └── image1.jpg
```

Each subfolder represents a different class label.

---

## 🧪 Workflow

1. **Feature Extraction**: Histogram equalization + HOG descriptors.
2. **Preprocessing**: PCA and scaling.
3. **Training**: SVM with grid search.
4. **Evaluation**: Accuracy, classification report, confusion matrix.
5. **OCR**: MSER-based text region detection and text extraction using Tesseract.
6. **Visualization**: Displays each preprocessing stage and predictions.

---

## 📈 Output

- **Accuracy metrics** for classification.
- **Detected text regions** and extracted text from sample images.
- **Visualization plots** for segmentation, filters, and classification results.

---

## 📌 Example Usage

```python
# Run end-to-end pipeline
image_path = '/path/to/your/image.jpg'
prediction, extracted_text = pipeline(image_path)
print("Prediction:", prediction)
print("Text:", extracted_text)
```

---

## 📚 References

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)

---

## 🧑‍💻 Author

*Developed as part of a minor project Image Processing and Text Extraction.*
