import cv2


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
