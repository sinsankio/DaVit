import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DaVit drive-server configuration")
    parser.add_argument('--max_speed', type=int, help='Maximum speed')
    parser.add_argument('--min_speed', type=int, help='Minimum speed')
    parser.add_argument('--host', type=str, default='', help='Host address')
    parser.add_argument('--port', type=int, default=4567, help='Port number')
    parser.add_argument('--validation', action='store_true', help='Enable validation mode')
    
    return parser.parse_args()

args = parse_args()

from datetime import datetime

import socketio
import eventlet
from flask import Flask

from utils import (
    decode_base64_image,
    preprocess_image,
    load_inference_engine,
    make_inference,
    calculate_throttle,
    smooth_steering,
    record_telemetry_results,
    create_driving_video
)

app = Flask(__name__)
sio = socketio.Server()

# Speed logic: if only one is specified, set both to that value; else use defaults
if args.max_speed is not None and args.min_speed is not None:
    MAX_SPEED, MIN_SPEED = args.max_speed, args.min_speed
elif args.max_speed is not None:
    MAX_SPEED = MIN_SPEED = args.max_speed
elif args.min_speed is not None:
    MAX_SPEED = MIN_SPEED = args.min_speed
else:
    MAX_SPEED, MIN_SPEED = 30, 5

HOST = args.host
PORT = args.port
VALIDATION_MODE = args.validation

current_speed = 0
speed_limit = MAX_SPEED
prev_steering = None

if VALIDATION_MODE:
    results_collected = []

def send_control_command(steering, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering),
            'throttle': str(throttle)
        },
        skip_sid=True
    )   

@sio.on('connect')
def connect(sid, environ):
    print("Client connected:", sid)
    send_control_command(0, 1)

@sio.on('*')
def catch_all(event, sid, data):
    print(f"Unhandled event '{event}' from client {sid}: {data}")

@sio.on('disconnect')
def disconnect(sid):
    print("Client disconnected:", sid)

@sio.on('telemetry')
def telemetry(sid, data):
    global current_speed, speed_limit, prev_steering

    if data:
        try:
            current_speed = float(data['speed'])
            if current_speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED

            image_base64 = data['image']
            image_decoded = decode_base64_image(image_base64)
            if image_decoded is None:
                send_control_command(0, 0) # safe fallback
                return

            image = preprocess_image(image_decoded)
            steering_angle = make_inference(image)
            steering_angle = smooth_steering(steering_angle, prev_steering)
            prev_steering = steering_angle

            throttle = calculate_throttle(steering_angle, current_speed, speed_limit)
    
            send_control_command(steering_angle, throttle)
            print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {current_speed:.2f}, Speed Limit: {speed_limit}")
        
            if VALIDATION_MODE:
                result = {
                    'telemetry': {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'image':image_decoded
                    },
                    'steering_angle': steering_angle,
                    'throttle': throttle,
                    'speed': current_speed    
                }
                results_collected.append(result)
        except Exception as e:
            print(f"Error processing telemetry data: {e}")

def main():
    load_inference_engine()

    print("Starting server...")
    app_with_sio = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen((HOST, PORT)), app_with_sio)

if __name__ == '__main__':
    main()
    if VALIDATION_MODE and results_collected:
        print('Collecting results...')

        save_dir = record_telemetry_results(results_collected)
        create_driving_video(save_dir)

        print('Results collected and saved.')
