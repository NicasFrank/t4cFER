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


class FERModel:
    def __init__(self, allow_gpu=True):
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if allow_gpu else ['CPUExecutionProvider']
        self.detection = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                                      providers=provider)
        self.detection.prepare(ctx_id=0, det_thresh=0.65, det_size=(640, 480))
        self.classification = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

    def infer_emotion(self, frame):
        faces = self.detection.get(frame)
        if faces:
            return self.__get_emotions(frame, faces)
        return None, None, None

    def __get_emotions(self, frame, faces):
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            face_img = frame[y1:y2, x1:x2, :]
            emotion, scores = self.classification.predict_emotions(face_img, logits=False)
            return emotion, scores, box


class FERView(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.presenter = FERPresenter()

        self.wm_title("Tech4compFER")
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.panel = None
        self.record_button = tk.Button(self, text="Start Recording", command=self.record_pressed)
        self.record_button.pack(side="bottom", fill="both", pady=10, padx=10)
        self.protocol("WM_DELETE_WINDOW", self.close_application)

        self.load_frame()

    def load_frame(self):
        if not self.presenter.img_queue.empty():
            image = ImageTk.PhotoImage(self.presenter.img_queue.get())
            if self.panel is None:
                self.panel = tk.Label(self.container, image=image)
                self.panel.image = image
                self.panel.pack(side="top")
            else:
                self.panel.configure(image=image)
                self.panel.image = image
        self.after(5, self.load_frame)

    def record_pressed(self):
        if self.presenter.recording:
            self.record_button.config(text="Start Recording")
        else:
            self.record_button.config(text="Stop Recording")
        self.presenter.switch_recording()

    def close_application(self):
        self.presenter.release()
        self.destroy()


class FERPresenter:
    def __init__(self):
        self.img_queue = queue.Queue()
        self.recording = False
        self.model = FERModel()
        self.__vc = cv2.VideoCapture(0)
        self.worker_thread = threading.Thread(target=self.__update_frame)
        self.worker_thread.start()

    def __update_frame(self):
        while not self.recording:
            start_time = time.time()
            _, frame = self.__vc.read()
            frame_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_draw)
            emotion, _, box = self.model.infer_emotion(frame)
            if box is not None:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                draw.text(box.tolist(), emotion)
            fps = 1.0 / (time.time() - start_time)
            draw.text((0, 0), str(int(fps)))
            self.img_queue.put(frame_draw)
        return

    def __record_emotions(self):
        with open(datetime.now().strftime("%d_%m_%Y %Hh%Mm%Ss") + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            while self.recording:
                start_time = time.time()
                _, frame = self.__vc.read()
                _, emotion_values, _ = self.model.infer_emotion(frame)
                if emotion_values is not None:
                    writer.writerow([time.time()] + [*emotion_values])
                elapsed_time = time.time() - start_time
                time.sleep(max(float(0), 0.1 - elapsed_time))
            return

    def switch_recording(self):
        self.recording = not self.recording
        self.worker_thread.join()
        if self.recording:
            self.worker_thread = threading.Thread(target=self.__record_emotions)
        else:
            self.worker_thread = threading.Thread(target=self.__update_frame)
        self.worker_thread.start()

    def release(self):
        self.recording = not self.recording
        self.worker_thread.join()
        self.__vc.release()


if __name__ == '__main__':
    FERView().mainloop()
