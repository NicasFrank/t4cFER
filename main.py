import cv2
import threading
import tkinter as tk
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from PIL import Image, ImageDraw, ImageTk


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

    def __get_emotions(self, frame, faces):
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            face_img = frame[y1:y2, x1:x2, :]
            emotion, scores = self.classification.predict_emotions(face_img, logits=False)
            return emotion, scores


class GUI(tk.Tk):
    def __init__(self, app):
        tk.Tk.__init__(self)
        self.wm_title("Tech4compFER")
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.panel = None
        self.record_button = tk.Button(self, text="Start Recording", command=app.record_emotions)
        self.record_button.pack(side="bottom", fill="both", pady=10, padx=10)

    def update_frame(self, frame):
        image = ImageTk.PhotoImage(frame)
        if self.panel is None:
            self.panel = tk.Label(self.container, image=image)
            self.panel.image = image
            self.panel.pack(side="top")
        else:
            self.panel.configure(image=image)
            self.panel.image = image


class Tech4compFER:
    def __init__(self, vc):
        self.gui = GUI(self)
        self.fer = FER()
        self.vc = vc
        self.show_gui = True
        self.gui_thread = threading.Thread(target=self.draw_gui)
        self.gui_thread.start()
        self.gui.mainloop()

    def draw_gui(self):
        while self.show_gui:
            _, frame = self.vc.read()
            self.gui.update_frame(self.fer.visualize(frame))
        self.gui.withdraw()
        return

    def record_emotions(self):
        self.show_gui = False


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)

    Tech4compFER(video_capture)

    video_capture.release()
