'''
Camera interface to start the TCP camera thread and receive frames from the camera server.
'''

import socket
import struct
import threading
import time
import cv2
import numpy as np

class Camera:
    """Camera interface to start the TCP camera thread"""
    def __init__(self, host='127.0.0.1', port=5599):
        self.host = host
        self.port = port
        self.thread_stop_event = None
        self.camera_thread = None


    def start_thread(self, callback):
        """Start the TCP camera thread"""
        if self.thread_stop_event is not None:
            return  # already running
        self.thread_stop_event = threading.Event()  # init BEFORE starting thread

        def camera_thread():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.host, self.port))
                    print(f"Connected to camera at {self.host}:{self.port}")
                    while not self.thread_stop_event.is_set():
                        frame = self.get_frame(s)
                        if frame is not None:
                            callback(frame)
                        else:
                            print("Failed to receive frame")
                            break
            except Exception as e:
                print(f"Camera thread error: {e}")
            finally:
                self.thread_stop_event.set()

        self.camera_thread = threading.Thread(target=camera_thread, daemon=True)
        self.camera_thread.start()

    def stop_thread(self):
        """Stop the TCP camera thread"""
        if self.thread_stop_event is not None:
            self.thread_stop_event.set()
            if self.camera_thread is not None:
                self.camera_thread.join(timeout=2)
            self.thread_stop_event = None
            self.camera_thread = None

    def get_frame(self, s):
        """Get the latest camera frame.
        Webots camera stream is RGB (3 bytes/pixel).
        Converts to BGR so OpenCV display conventions remain unchanged.
        """
        # 1. Read Header (4 bytes: Width, Height)
        header_data = self._recv_all(s, 4)
        if not header_data: return None
        width, height = struct.unpack("=HH", header_data)

        # 2. Read RGB image data (3 bytes per pixel)
        img_data = self._recv_all(s, width * height * 3)
        if not img_data: return None

        # Convert RGB -> BGR for OpenCV
        rgb = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return frame


    def is_running(self):
        return self.thread_stop_event is not None and not self.thread_stop_event.is_set()

    def _recv_all(self, sock, n):
        """Helper to receive exactly n bytes from a TCP socket"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return data
    
    def __del__(self):
        """Destructor to ensure threads are stopped"""
        self.stop_thread()