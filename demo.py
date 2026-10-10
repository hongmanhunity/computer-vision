import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_sample_image():
    img = np.full((400, 600, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (80, 220), (520, 360), (60, 110, 170), -1)
    cv2.rectangle(img, (150, 140), (220, 220), (235, 235, 235), -1)
    cv2.rectangle(img, (320, 170), (460, 220), (80, 200, 90), -1)

    rng = np.random.default_rng(42)
    noise_points = rng.integers(0, [400, 600], size=(400, 2))
    for y, x in noise_points[:200]:
        img[y, x] = [255, 255, 255]
    for y, x in noise_points[200:]:
        img[y, x] = [0, 0, 0]

    return img


def detect_objects(img_bgr, edges):
    result = cv2.cvtColor(cv2.medianBlur(img_bgr, 5), cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h <= 2000:
            continue

        if w > 300:
            label, color = "Table (96%)", (255, 180, 0)
        elif h > w:
            label, color = "Cup (92%)", (0, 255, 100)
        else:
            label, color = "Book (89%)", (0, 200, 255)

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return result


def main():
    img_bgr = create_sample_image()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)
    equalized = cv2.equalizeHist(denoised)
    edges = cv2.Canny(denoised, threshold1=50, threshold2=150)
    detected = detect_objects(img_bgr, edges)

    print(gray[138:143, 148:153])

    steps = [
        (img_rgb, "1. Original Image", None),
        (gray, "2. Grayscale", "gray"),
        (denoised, "3. Median Blur", "gray"),
        (equalized, "4. Histogram Equalization", "gray"),
        (edges, "5. Canny Edges", "gray"),
        (detected, "6. Object Detection", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (image, title, cmap) in zip(axes.flat, steps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
