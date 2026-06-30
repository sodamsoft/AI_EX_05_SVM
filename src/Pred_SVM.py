import os
import cv2
import numpy as np
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 28
MODEL_PATH = os.path.join(current_dir, "svm_digit_model.pkl")
IMAGE_PATH = os.path.join(current_dir, "data\digit.jpg")
OUTPUT_PATH = os.path.join(current_dir, "data\digit_result.jpg")


def preprocess_image(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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


def find_digit_boxes(roi_img):
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # 너무 작은 잡음 제거
        if w < 5 or h < 10:
            continue

        # 너무 큰 영역 제거
        if w > roi_img.shape[1] * 0.8 or h > roi_img.shape[0] * 0.95:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes


def predict_digits_in_roi(model, roi_img):
    boxes = find_digit_boxes(roi_img)

    predictions = []
    for i, (x, y, w, h) in enumerate(boxes):
        margin = 4
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(roi_img.shape[1], x + w + margin)
        y2 = min(roi_img.shape[0], y + h + margin)

        digit_crop = roi_img[y1:y2, x1:x2]
        processed = preprocess_image(digit_crop)

        # 디버그 저장
        debug_big = cv2.resize(processed, (280, 280), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f"debug_digit_{i}.png", debug_big)

        features = processed.flatten().astype(np.float32) / 255.0
        features = features.reshape(1, -1)

        pred = model.predict(features)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = np.max(model.predict_proba(features))

        predictions.append({
            "box": (x1, y1, x2 - x1, y2 - y1),
            "digit": int(pred),
            "prob": float(prob) if prob is not None else None
        })

    return predictions



def main():
    print("시작")
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없습니다: {MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"이미지 파일이 없습니다: {IMAGE_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    print("모델 로드 완료")

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("이미지를 읽을 수 없습니다.")
        return

    # -----------------------------
    # 숫자 영역(ROI) 지정
    # -----------------------------
    height, width = image.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, width, height
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()

    preds = predict_digits_in_roi(model, roi)

    cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    digit_text = ""

    for p in preds:
        x, y, w, h = p["box"]
        digit = p["digit"]
        prob = p["prob"]

        gx1 = roi_x1 + x
        gy1 = roi_y1 + y
        gx2 = roi_x1 + x + w
        gy2 = roi_y1 + y + h

        cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

        label = f"{digit}"
        color = (0, 0, 255)
        if prob is not None:
            label += f" ({prob:.2f})"     

        cv2.putText(
            image,
            label,
            (gx1, gy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        digit_text += str(digit)
    '''
    if digit_text:
        cv2.putText(
            image,
            f"Predicted: {digit_text}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
    '''
    
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"예측 결과: {digit_text}")
    print(f"결과 이미지 저장 완료: {OUTPUT_PATH}")

    cv2.imshow("Calendar Digit Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
