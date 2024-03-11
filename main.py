import queue
import csv
import cv2
import time
import threading
import tkinter as tk
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from PIL import Image, ImageDraw, ImageTk
from datetime import datetime


class FER:
    def __init__(self, allow_gpu=True):
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if allow_gpu else ['CPUExecutionProvider']
        self.detection = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                                      providers=provider)
        self.detection.prepare(ctx_id=0, det_size=(640, 640))
        self.classification = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

    def visualize(self, frame):
        faces = self.detection.get(frame)
        frame_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_draw)
        if faces:
            emotion, _ = self.__get_emotions(frame, faces)
            for face in faces:
                box = face.bbox.astype(int)
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                draw.text(box.tolist(), emotion)
        return frame_draw

    def infer_emotion(self, frame):
        faces = self.detection.get(frame)
        if faces:
            return self.__get_emotions(frame, faces)
        return None, None

    def __get_emotions(self, frame, faces):
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            face_img = frame[y1:y2, x1:x2, :]
            emotion, scores = self.classification.predict_emotions(face_img, logits=False)
            return emotion, scores


class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.img_queue = queue.Queue()
        self.logic = Tech4compFER(self.img_queue)

        self.wm_title("Tech4compFER")
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.panel = None
        self.record_button = tk.Button(self, text="Start Recording", command=self.record_pressed)
        self.record_button.pack(side="bottom", fill="both", pady=10, padx=10)
        self.protocol("WM_DELETE_WINDOW", self.close_application)

        self.update_frame()

    def update_frame(self):
        if not self.img_queue.empty():
            image = ImageTk.PhotoImage(self.img_queue.get())
            if self.panel is None:
                self.panel = tk.Label(self.container, image=image)
                self.panel.image = image
                self.panel.pack(side="top")
            else:
                self.panel.configure(image=image)
                self.panel.image = image
        self.after(5, self.update_frame)

    def record_pressed(self):
        if self.logic.recording:
            self.record_button.config(text="Start Recording")
            self.logic.stop_recording()
        else:
            self.record_button.config(text="Stop Recording")
            self.logic.start_recording()

    def close_application(self):
        self.logic.recording = not self.logic.recording
        self.logic.worker_thread.join()
        self.logic.vc.release()
        self.destroy()


class Tech4compFER:
    def __init__(self, img_queue):
        self.img_queue = img_queue
        self.recording = False
        self.fer = FER()
        self.vc = cv2.VideoCapture(0)
        self.worker_thread = threading.Thread(target=self.update_frames)
        self.worker_thread.start()

    def update_frames(self):
        while not self.recording:
            _, frame = self.vc.read()
            self.img_queue.put(self.fer.visualize(frame))
        return

    def record_emotions(self):
        with open(datetime.now().strftime("%d_%m_%Y %Hh%Mm%Ss") + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            while self.recording:
                start_time = time.time()
                _, frame = self.vc.read()
                _, emotion_values = self.fer.infer_emotion(frame)
                if emotion_values is not None:
                    writer.writerow([time.time()] + [*emotion_values])
                elapsed_time = time.time() - start_time
                time.sleep(max(float(0), 0.1 - elapsed_time))
            return

    def start_recording(self):
        self.recording = True
        self.worker_thread.join()
        self.worker_thread = threading.Thread(target=self.record_emotions)
        self.worker_thread.start()

    def stop_recording(self):
        self.recording = False
        self.worker_thread.join()
        self.worker_thread = threading.Thread(target=self.update_frames)
        self.worker_thread.start()


if __name__ == '__main__':

    GUI().mainloop()

