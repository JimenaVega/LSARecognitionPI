import cv2
import time
import json

def run_cam_performance_test(cam_width, cam_height, cam_fps):
    print('Opening camera...')
    cam = cv2.VideoCapture(0)
    print('Configuring camera...')
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cam.set(cv2.CAP_PROP_FPS, cam_fps)
    print('Camera ready.')

    frames_counter = 0
    t_start = time.time()

    for i in range(120):
        ret, frame = cam.read()
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        print(f'Frame size: {frame_width}x{frame_height} - Cam config: {cam_width}x{cam_height}')
        frames_counter += 1

    elapsed_time = time.time() - t_start
    real_fps = int(frames_counter / elapsed_time)

    cam.release()

    return dict({'frames_read': frames_counter,
                 'elapsed_time': round(elapsed_time, 2),
                 'real_fps': real_fps})

cam_config_sets = [[160, 120, 10], [160, 120, 20], [160, 120, 30],
                   [176, 144, 10], [176, 144, 20], [176, 144, 30],
                   [320, 240, 10], [320, 240, 20], [320, 240, 30],
                   [352, 288, 10], [352, 288, 20], [352, 288, 30],
                   [640, 360, 10], [640, 360, 20], [640, 360, 30],
                   [640, 480, 10], [640, 480, 20], [640, 480, 30],
                   [800, 600, 10], [800, 600, 20], [800, 600, 30],
                   [848, 480, 10], [848, 480, 20], [848, 480, 30],
                   [960, 540, 10], [960, 540, 20], [960, 540, 30],
                   [1024, 576, 10], [1024, 576, 20], [1024, 576, 30],
                   [1280, 720, 10], [1280, 720, 20], [1280, 720, 30],
                   [1920, 1080, 10], [1920, 1080, 20], [1920, 1080, 30],]

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

