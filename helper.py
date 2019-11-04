import cv2
from matplotlib import pyplot as plt


def convert(frame, src_model="bgr", dest_model="hls"):

    if src_model == "bgr" and dest_model == "hsv":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif src_model == "bgr" and dest_model == "hls":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    elif src_model == "bgr" and dest_model == "yuv":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    elif src_model == "bgr" and dest_model == "ycrcb":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    elif src_model == "bgr" and dest_model == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif src_model == "hsv" and dest_model == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    elif src_model == "hls" and dest_model == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_HLS2RGB)
    elif src_model == "yuv" and dest_model == "yuv":
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
    elif src_model == "ycrcb" and dest_model == "ycrcb":
        frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2RGB)
    else:
        raise Exception("ERROR:", "src_model or dest_model not implemented")

    return frame


def sliding_window(img, step_size, window_size):
    height, width, channel = img.shape
    window_width, window_height = window_size
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            frame = img[y : y + window_height, x : x + window_width]
            frame_height, frame_width, channel = frame.shape
            x_min, y_min = x, y

            # if frame_height != window_height:
            #     y_min = height - window_height
            # if frame_width != window_width:
            #     x_min = width - window_width
            # frame = img[y_min : y + window_height, x_min : x_min + window_width]
            if frame_height != window_height or frame_width != window_width:
                continue

            yield (x_min, y_min, frame)


def show_images(imgs, per_row=3, per_col=2, W=10, H=5, tdpi=80):
    fig, ax = plt.subplots(per_col, per_row, figsize=(W, H), dpi=tdpi)
    ax = ax.ravel()

    for i in range(len(imgs)):
        img = imgs[i]
        ax[i].imshow(img)

    for i in range(per_row * per_col):
        ax[i].axis("off")

