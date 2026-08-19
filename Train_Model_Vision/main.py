# Usage:
#   pip install -r requirements.txt
#   python main.py image --path path/to/image.jpg
#   python main.py video --path path/to/video.mp4 --output result.mp4
#   python main.py webcam [--camera 0]

import cv2                      # for reading images/video/webcam frames and drawing boxes
import torch                    # pytorch core
from ultralytics import YOLO    # ultralytics wrapper, works with yolov5/yolov8 .pt models
import argparse                 # to read command line arguments
import os                       # to check file paths

# path to the model file, sits next to this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

# pick the best available device: nvidia gpu (cuda), apple silicon gpu (mps), otherwise cpu
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def load_model():
    # load the .pt weights into a YOLO model object
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    return model


def detect_image(model, image_path):
    # read the image from disk
    frame = cv2.imread(image_path)

    # run inference on a single frame
    results = model(frame, device=DEVICE)[0]

    # draw the boxes on top of the frame
    annotated = results.plot()

    # show result in a window until a key is pressed
    cv2.imshow("Detection - Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video(model, video_path, output_path=None):
    # open the video file
    cap = cv2.VideoCapture(video_path)

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break  # video ended

        # run inference on the current frame
        results = model(frame, device=DEVICE)[0]

        # draw detection boxes on the frame
        annotated = results.plot()

        # show the annotated frame
        cv2.imshow("Detection - Video", annotated)

        # write the annotated frame to the output file, if requested
        if writer is not None:
            writer.write(annotated)

        # press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def detect_webcam(model, camera_index=0, output_path=None):
    # open the default camera (or the index passed in)
    cap = cv2.VideoCapture(camera_index)

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # camera stopped sending frames

        # run inference on the live frame
        results = model(frame, device=DEVICE)[0]

        # draw detection boxes on the frame
        annotated = results.plot()

        # show the live annotated feed
        cv2.imshow("Detection - Live Camera", annotated)

        # write the annotated frame to the output file, if requested
        if writer is not None:
            writer.write(annotated)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # simple CLI: choose the source type and optional file path
    parser = argparse.ArgumentParser(description="Run YOLO detection on image, video or webcam")
    parser.add_argument("mode", choices=["image", "video", "webcam"], help="type of input to run detection on")
    parser.add_argument("--path", help="path to the image or video file (not needed for webcam)")
    parser.add_argument("--camera", type=int, default=0, help="camera index for webcam mode")
    parser.add_argument("--output", help="path to save the annotated output as a video file (e.g. out.mp4)")
    args = parser.parse_args()

    # load the model once and reuse it for whichever mode was chosen
    yolo_model = load_model()

    if args.mode == "image":
        detect_image(yolo_model, args.path)
    elif args.mode == "video":
        detect_video(yolo_model, args.path, args.output)
    elif args.mode == "webcam":
        detect_webcam(yolo_model, args.camera, args.output)
