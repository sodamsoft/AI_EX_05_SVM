import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 28

def preprocess_image(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU    #_INV
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        xs, ys, xe, ye = [], [], [], []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)

        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xe), max(ye)

        roi = thresh[y1:y2, x1:x2]

        h, w = roi.shape
        if w > h:
            new_w = 20
            new_h = int((h / w) * 20)
        else:
            new_h = 20
            new_w = int((w / h) * 20)

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        padded = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        x_offset = (IMG_SIZE - new_w) // 2
        y_offset = (IMG_SIZE - new_h) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi

        return padded
    else:
        return cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))

def load_data(dataset_path):
    X = []
    y = []

    print(f"Loading data from {dataset_path}...")
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp")

    for label in range(10):
        folder_path = os.path.join(dataset_path, str(label))
        if not os.path.isdir(folder_path):
            print(f"Warning: Directory not found - {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(valid_ext):
                continue

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                processed_img = preprocess_image(img)
                features = processed_img.flatten().astype(np.float32) / 255.0
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    DATASET_PATH = os.path.join(current_dir, "data")
    MODEL_SAVE_PATH = os.path.join(current_dir, "svm_digit_model.pkl")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: '{DATASET_PATH}' 폴더가 없습니다. 0~9 폴더를 이 폴더 안에 넣어주세요.")
        exit(1)

    X, y = load_data(DATASET_PATH)

    if len(X) == 0:
        print("학습할 데이터가 없습니다.")
        exit(1)

    print(f"총 {len(X)}개의 이미지를 불러왔습니다. 학습을 시작합니다...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel='rbf', C=5, gamma=0.05, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"검증 세트 정확도 (Accuracy): {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"학습 완료! 모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")
