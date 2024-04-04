import cv2
import time
import json

def run_cam_performance_test(cam_width, cam_height, cam_fps):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cam.set(cv2.CAP_PROP_FPS, cam_fps)

    frames_counter = 0
    t_start = time.time()

    for i in range(240):
        ret, frame = cam.read()
        print(f'Frame size: {len(frame)} - Cam config: {cam_width}x{cam_height}')
        frames_counter += 1

    elapsed_time = time.time() - t_start
    real_fps = int(frames_counter / elapsed_time)


    cam.release()

    return dict({'frames_read': frames_counter,
                 'elapsed_time': round(elapsed_time, 2),
                 'real_fps': real_fps})

cam_config_sets = [[320, 240, 5], [320, 240, 10], [320, 240, 15], [320, 240, 20], [320, 240, 25], [320, 240, 30], [320, 240, 60],
                   [640, 480, 5], [640, 480, 10], [640, 480, 15], [640, 480, 20], [640, 480, 25], [640, 480, 30], [640, 480, 60],
                   [720, 480, 5], [720, 480, 10], [720, 480, 15], [720, 480, 20], [720, 480, 25], [720, 480, 30], [720, 480, 60],
                   [800, 600, 5], [800, 600, 10], [800, 600, 15], [800, 600, 20], [800, 600, 25], [800, 600, 30], [800, 600, 60],
                   [1280, 720, 5], [1280, 720, 10], [1280, 720, 15], [1280, 720, 20], [1280, 720, 25], [1280, 720, 30], [1280, 720, 60],
                   [1280, 960, 5], [1280, 960, 10], [1280, 960, 15], [1280, 960, 20], [1280, 960, 25], [1280, 960, 30], [1280, 960, 60],
                   [1920, 1080, 5], [1920, 1080, 10], [1920, 1080, 15], [1920, 1080, 20], [1920, 1080, 25], [1920, 1080, 30], [1920, 1080, 60],]

performance_results = dict()
n_test = 1
test_start = time.time()

for cam_config in cam_config_sets:
    print(f'Running test {n_test}/{len(cam_config_sets)}...')
    res = run_cam_performance_test(cam_width=cam_config[0],
                                   cam_height=cam_config[1],
                                   cam_fps=cam_config[2])
    
    performance_results[f'{cam_config[0]}x{cam_config[1]}x{cam_config[2]}'] = res

    print(f'Test {n_test}/{len(cam_config_sets)} completed.')

    n_test += 1

print(f'All tests completed in {time.time() - test_start} seconds.')

with open("cam_performance_results.json", "w") as outfile: 
    json.dump(performance_results, outfile)

