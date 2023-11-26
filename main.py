import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    # model_path = f'{os.getcwd()}/pose_landmarker_lite.task'
    # model_path = f'{os.getcwd()}/pose_landmarker_full.task'
    model_path = f'{os.getcwd()}/pose_landmarker_heavy.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    
    image_path = f'{os.getcwd()}/ollie1.jpg'
    img = cv2.imread(image_path)
    img_height = img.shape[0]
    img_width = img.shape[1]    

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)
        pose_landmarker_result = landmarker.detect(mp_image)
        landmarks = pose_landmarker_result.pose_landmarks[0]
        important_landmarks = [25, 26, 27, 28]
        for L in important_landmarks:
            cur_landmark = landmarks[L]
            center_x = int(img_width * cur_landmark.x)
            center_y = int(img_height * cur_landmark.y)
            cv2.circle(img, (center_x, center_y), 10, (255, 0, 0) , 5)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if (__name__ == "__main__"):
    main()