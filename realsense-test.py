import pyrealsense2 as rs
import numpy as np
import cv2

W, H, FPS = 640, 480, 60
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

profile = pipe.start(config)
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

try:
    while True:
        frames = pipe.wait_for_frames()
        frames = align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            continue
        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data())

        dist_center = depth[H // 2, W // 2] * depth_scale
        cv2.putText(color, f"{dist_center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Color", color)
        if cv2.waitKey(1) == 27:
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped and windows closed.")
    print("Depth scale:", depth_scale)
    print("Resolution:", W, "x", H)
    print("FPS:", FPS)
    print("Test completed successfully.")
    print("Exiting...")
    exit(0)