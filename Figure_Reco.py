import cv2
import numpy as np

# Create a VideoCapture object for the default camera (0) or specify the camera's index if you have multiple cameras.
cap = cv2.VideoCapture(0)

def process():
    while True:
        ret, frame = cap.read()  # Read a frame from the camera

        frame = frame[int(frame.shape[0] * 0.25):int(frame.shape[0] * 0.75), int(frame.shape[1] * 0.66):]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 5, 75, 75)
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(max(255, (1.0 + sigma) * v))
        edges = cv2.Canny(blurred, lower, upper)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        output_gray = None

        if len(contours) > 0:
            big_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(big_contour)
            peri = cv2.arcLength(big_contour, True)
            vertices = cv2.approxPolyDP(big_contour, 0.01 * peri, True)
            num_vertices = len(vertices)

            x, y, w, h = cv2.boundingRect(big_contour)
            src = np.float32([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
            dst = np.float32([(0, 0), (0, 690), (690, 690), (690, 0)])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            matrix = cv2.getPerspectiveTransform(src, dst)
            output_image = cv2.warpPerspective(frame, matrix, (690, 690), flags=cv2.INTER_LINEAR)
            output_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

        if output_gray is not None:
            cv2.imshow("Processed Frame", output_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process()
