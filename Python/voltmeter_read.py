# -*- coding: utf-8 -*-
"""
Analog Voltmeter Reader - Core Logic
AI(YOLOv8)による計器検出と、画像処理による針の角度解析を行います。
"""

import os
import sys
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

# =================================================================
# 1. パス・環境設定
# =================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "Images")
output_dir = os.path.join(script_dir, "outputs")
model_path = os.path.join(script_dir, "best.pt")

# =================================================================
# 2. YOLOによる電圧計の検出と回転補正
# =================================================================
def detect_and_straighten_yolo(image_path, output_bbox_path, output_cropped_path):
    """
    YOLOv8を使用して画像から電圧計を検出し、
    セグメンテーション情報に基づいて水平に補正した画像を保存します。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像が見つかりません: {image_path}")

    model = YOLO(model_path)
    # 低信頼度でも検出を試みる（conf=0.01）
    results = model.predict(image, conf=0.01, verbose=False)

    if not results or len(results[0].boxes) == 0:
        cv2.imwrite(output_bbox_path, image)
        print("Failed: 電圧計が検出されませんでした。", file=sys.stdout, flush=True)
        return None, ""

    result = results[0]
    box = result.boxes[0]
    confidence = f"{box.conf.item():.2f}"

    # --- セグメンテーションマスクがある場合の回転補正処理 ---
    if result.masks is not None and len(result.masks.xy) > 0:
        mask = result.masks.xy[0]
        pts = np.array(mask, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        center, (w, h), angle = rect

        # 角度補正（上向きを基準に統一）
        corrected_angle = angle if w < h else angle + 90
        if corrected_angle < -45:
            corrected_angle += 90
        elif corrected_angle > 45:
            corrected_angle -= 90

        # アフィン変換による回転
        M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 回転後の画像からクロップ
        w, h = int(w), int(h)
        rotated_box = np.intp(cv2.boxPoints(((center[0], center[1]), (w, h), corrected_angle)))
        x_min, y_min = np.min(rotated_box, axis=0)
        x_max, y_max = np.max(rotated_box, axis=0)
        cropped = rotated[max(0, y_min):y_max, max(0, x_min):x_max]

        # 結果の保存
        cv2.imwrite(output_bbox_path, result.plot())
        cv2.imwrite(output_cropped_path, cropped)
        return cropped, confidence
    
    # --- マスクがない場合は通常の矩形クロップ ---
    else:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        cv2.imwrite(output_bbox_path, result.plot())
        cv2.imwrite(output_cropped_path, cropped)
        return cropped, confidence

# =================================================================
# 3. 針の角度取得ロジック（赤色抽出 + 輪郭検出）
# =================================================================
def get_needle_angle(img_or_path):
    """
    画像内の赤い針を検出し、その中心点からの角度を計算します。
    """
    img = img_or_path if isinstance(img_or_path, np.ndarray) else cv2.imread(img_or_path)
    if img is None:
        raise ValueError("入力画像が空です。")

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # ノイズ除去
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 赤色の色相範囲を2パターンで抽出
    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([20, 255, 255])
    lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # モルフォロジー演算によるノイズ消去と強調
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("針が見つかりません。")
    
    # 最大面積の輪郭（＝針）を特定
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = np.intp(cv2.boxPoints(rect))

    # 中心から最も遠い点を「針の先端」とする
    center_pt = np.array([cx, cy])
    tip = tuple(box[np.argmax([np.linalg.norm(p - center_pt) for p in box])])

    # 角度計算
    angle = np.degrees(np.arctan2(cy - tip[1], tip[0] - cx))
    
    # デバッグ描画
    res_img = img.copy()
    cv2.line(res_img, (cx, cy), tip, (0, 0, 255), 2)
    cv2.circle(res_img, (cx, cy), 5, (0, 255, 0), -1)
    
    return angle, tip, res_img

# =================================================================
# 4. 実行エントリポイント
# =================================================================
if __name__ == "__main__":
    # --- A. テストモード（YOLO読み込み確認） ---
    if "--test" in sys.argv:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"モデルファイル無し: {model_path}")
            _ = YOLO(model_path)
            print(f"YOLOモデルが正常に読み込まれました: {ultralytics.__version__}", flush=True)
            sys.exit(0)
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    # --- B. 通常実行（解析処理） ---
    if len(sys.argv) < 4:
        print("Usage: python voltmeter_read.py <input> <bbox_out> <crop_out>", file=sys.stderr)
        sys.exit(1)

    image_path, output_bbox_path, output_cropped_path = sys.argv[1:4]
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 電圧計の切り出しと補正
        cropped_img, conf = detect_and_straighten_yolo(image_path, output_bbox_path, output_cropped_path)
        if cropped_img is None: sys.exit(1)

        # キャリブレーション用画像の角度取得（0Vと3Vの基準点）
        angle_0v, _, _ = get_needle_angle(os.path.join(images_dir, "cropped_0.0v_t1.png"))
        angle_3v, _, _ = get_needle_angle(os.path.join(images_dir, "cropped_3.0v_t1.png"))
        
        # 測定画像の解析
        n_angle, n_tip, final_img = get_needle_angle(cropped_img)

        # 線形補間による電圧値計算
        voltage = round(0 + (n_angle - angle_0v) * (3 - 0) / (angle_3v - angle_0v), 1)

        # 結果の描画と出力
        cv2.putText(final_img, f"{voltage:.1f} V", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_cropped_path, final_img)

        print("Success", flush=True)
        print(voltage, flush=True)
        print(os.path.abspath(output_bbox_path), flush=True)
        print(os.path.abspath(output_cropped_path), flush=True)
        print(conf, flush=True)

    except Exception as e:
        print(f"Failed: {e}", file=sys.stdout, flush=True)
        sys.exit(1)
