import cv2
import numpy as np
import os


def put_text_in_frame(frame, text, location="top-right", bg="glassy"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    h, w = frame.shape[:2]

    frame = frame.copy()
    # Get text sizes
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    if location == "top-right":
        p_wmin, p_wmax = w - text_width - 20, w
        p_hmin, p_hmax = 0, text_height + 20
    else:
        p_wmin, p_wmax = 0, text_width + 20
        p_hmin, p_hmax = 0, text_height + 20

    t_x, t_y = p_wmin + 10, p_hmax - 10

    if bg == "glassy":
        color = (255, 255, 255)
        frame[p_hmin: p_hmax, p_wmin: p_wmax, :] = (frame[p_hmin: p_hmax, p_wmin: p_wmax, :].astype(np.float32) * 0.6).astype(np.uint8)
    else:
        color = (255, 255, 255)
        frame[p_hmin: p_hmax, p_wmin: p_wmax, :] = [0, 100, 100]

    cv2.putText(frame, text, (t_x, t_y), font,
                font_scale, color, thickness, cv2.LINE_AA)

    # if location == "top-right":
    #     frame[0: text_height + 20, w - text_width - 20:, :] = (frame[0: text_height + 20, w - text_width - 20:, :].astype(np.float32) * 0.6).astype(np.uint8)
    #     # Put the text
    #     cv2.putText(frame, text, (w - text_width - 10, text_height + 10), font,
    #                 font_scale, color, thickness, cv2.LINE_AA)
    # else:
    #     frame[0: text_height + 20, 0: text_width + 20] = (frame[0: text_height + 20, 0: text_width + 20].astype(np.float32) * 0.6).astype(np.uint8)
    #     # Put the text
    #     cv2.putText(frame, text, (10, text_height + 10), font,
    #                 font_scale, color, thickness, cv2.LINE_AA)

    return frame


# def put_text_in_frame(frame, text):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.0
#     thickness = 1
#     color = (255, 255, 255)
#
#     h, w = frame.shape[:2]
#
#     # Get text size
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#
#     # Padding around the text
#     pad_x, pad_y = 10, 5
#     box_w = text_width + 2 * pad_x
#     box_h = text_height + 2 * pad_y
#     x1 = w - box_w - 10
#     y1 = 10
#     x2 = x1 + box_w
#     y2 = y1 + box_h
#
#     # Ensure coordinates stay within the frame
#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(w, x2)
#     y2 = min(h, y2)
#
#     # Extract ROI and blur it
#     roi = frame[y1:y2, x1:x2].copy()
#     blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
#
#     # Blend original and blurred for frosted glass effect
#     overlay = cv2.addWeighted(roi, 0.3, blurred_roi, 0.7, 0)
#
#     # Put back the blurred region
#     frame[y1:y2, x1:x2] = overlay
#
#     # Put the text over the glassy region
#     text_x = x1 + pad_x
#     text_y = y1 + pad_y + text_height
#     cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
#
#     return frame


def viz_img_grid(data_grid, str_grid=None, spacing=10):

    nr = len(data_grid)
    nc = np.array([len(data_grid[i]) for i in range(nr)]).astype(np.int32).max()
    nc = len(data_grid[0])

    h, w = data_grid[0][0].shape[:2]

    frame_h = h * nr + spacing * (nr + 1)
    frame_w = w * nc + spacing * (nc + 1)

    viz_frame = np.ones((frame_h, frame_w, 3)).astype(np.uint8) * 255

    y_top_left, x_top_left = spacing, spacing
    for i in range(nr):
        if len(data_grid[i]) == nc:
            x_top_left = spacing
        else:
            num_frames = len(data_grid[i])
            empty_space = w * nc + (nc - 1) * spacing - w * num_frames - (num_frames - 1) * spacing
            x_top_left = spacing + empty_space // 2

        for j in range(len(data_grid[i])):
            curr_frame = data_grid[i][j]
            if str_grid is not None and str_grid[i][j] is not None:
                curr_frame = put_text_in_frame(curr_frame, str_grid[i][j])

            curr_frame = cv2.cvtColor(curr_frame.copy(), cv2.COLOR_RGB2BGR)


            viz_frame[y_top_left: y_top_left + h, x_top_left: x_top_left + w] = curr_frame
            x_top_left += (w + spacing)
        y_top_left += (h + spacing)

    return viz_frame


def save_frame(frame, output_path, convert_to_bgr=True):
    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(output_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if convert_to_bgr:
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

    # Write the image
    success = cv2.imwrite(output_path, frame)

    if not success:
        raise IOError(f"Failed to write image to {output_path}")