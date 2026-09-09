import sqlite3
import shutil
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FACES_DIR = DATA_DIR / "faces"
DB_PATH = DATA_DIR / "face_recognition.db"
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
LOCAL_CASCADE_PATH = DATA_DIR / "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"


class FaceDatabase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                confidence REAL NOT NULL,
                recognized_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            """
        )
        self.connection.commit()

    def add_user(self, name: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO users(name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat(timespec="seconds")),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def users(self):
        return self.connection.execute("SELECT * FROM users ORDER BY name").fetchall()

    def add_log(self, user_id: int | None, confidence: float):
        self.connection.execute(
            "INSERT INTO recognition_logs(user_id, confidence, recognized_at) VALUES (?, ?, ?)",
            (user_id, confidence, datetime.now().isoformat(timespec="seconds")),
        )
        self.connection.commit()

    def recent_logs(self, limit=100):
        return self.connection.execute(
            """
            SELECT recognition_logs.recognized_at, recognition_logs.confidence,
                   COALESCE(users.name, 'Người lạ') AS name
            FROM recognition_logs LEFT JOIN users ON users.id = recognition_logs.user_id
            ORDER BY recognition_logs.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()


class FaceRecognizer:
    def __init__(self, database: FaceDatabase):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "Thiếu opencv-contrib-python. Hãy chạy: pip install -r requirements.txt"
            )
        self.database = database
        cascade_path = CASCADE_PATH if CASCADE_PATH.exists() else LOCAL_CASCADE_PATH
        if not cascade_path.exists():
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(CASCADE_URL, LOCAL_CASCADE_PATH)
                cascade_path = LOCAL_CASCADE_PATH
            except Exception as error:
                raise RuntimeError(
                    "Không tải được bộ nhận diện khuôn mặt. Hãy kiểm tra kết nối mạng rồi chạy lại."
                ) from error
        readable_path = Path(tempfile.gettempdir()) / "face_recognition_haarcascade.xml"
        if cascade_path != readable_path:
            shutil.copyfile(cascade_path, readable_path)
        self.detector = cv2.CascadeClassifier(str(readable_path))
        if self.detector.empty():
            raise RuntimeError(
                f"Không thể đọc file Haar cascade: {readable_path}. "
                "Hãy cài lại dependency bằng pip install -r requirements.txt"
            )
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = (
            mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            if mp is not None and hasattr(mp, "solutions")
            else None
        )
        self.ready = False
        self.labels: dict[int, str] = {}
        self.threshold = 85.0
        self.train()

    def detect(self, gray):
        height, width = gray.shape[:2]
        result = (
            self.face_detector.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
            if self.face_detector is not None
            else None
        )
        if result and result.detections:
            boxes = []
            for detection in result.detections:
                box = detection.location_data.relative_bounding_box
                x = max(0, int(box.xmin * width))
                y = max(0, int(box.ymin * height))
                right = min(width, int((box.xmin + box.width) * width))
                bottom = min(height, int((box.ymin + box.height) * height))
                if right > x and bottom > y:
                    boxes.append((x, y, right - x, bottom - y))
            if boxes:
                return boxes
        normalized = cv2.equalizeHist(gray)
        detected = self.detector.detectMultiScale(
            normalized, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50)
        )
        return [tuple(map(int, box)) for box in detected]

    def train(self):
        images, labels = [], []
        self.labels = {int(row["id"]): row["name"] for row in self.database.users()}
        for user_id in self.labels:
            for image_path in (FACES_DIR / str(user_id)).glob("*.jpg"):
                image = self._read_image(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(user_id)
        if images:
            self.model.train(images, __import__("numpy").array(labels))
            self.ready = True

    @staticmethod
    def _read_image(path):
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None

    @staticmethod
    def _write_image(path, image):
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return False
        encoded.tofile(str(path))
        return path.exists() and path.stat().st_size > 0

    def recognize(self, gray, box):
        if not self.ready:
            return None, 0.0
        x, y, width, height = box
        face = gray[y : y + height, x : x + width]
        label, distance = self.model.predict(face)
        confidence = max(0.0, min(100.0, 100.0 - distance))
        if distance <= self.threshold and label in self.labels:
            return self.labels[label], confidence
        return None, confidence


class FaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nhận diện khuôn mặt thời gian thực")
        self.geometry("1120x720")
        self.minsize(960, 620)
        self.configure(bg="#101820")
        self.database = FaceDatabase(DB_PATH)
        self.recognizer = FaceRecognizer(self.database)
        self.camera = None
        self.running = False
        self.registering = False
        self.register_user_id = None
        self.register_count = 0
        self.last_log = {}
        self.current_image = None
        self.frame_job = None
        self.read_failures = 0
        self.pending_frame = None
        self._build_ui()
        self.refresh_tables()
        self.report_callback_exception = self._report_callback_exception
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _report_callback_exception(self, exception, value, traceback):
        self.stop_camera()
        self.status.config(text="Lỗi xử lý camera", fg="#ff8b7b")
        messagebox.showerror("Lỗi xử lý camera", f"{exception.__name__}: {value}")

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=8, font=("Segoe UI", 10))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        header = tk.Frame(self, bg="#101820", padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="FACE / LIVE", fg="#77e0c2", bg="#101820", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(header, text="Nhận diện khuôn mặt thời gian thực", fg="white", bg="#101820", font=("Segoe UI", 24, "bold")).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self, bg="#101820", padx=24)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#17252f")
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg="#f2f0e9", width=350, padx=18, pady=18)
        right.pack(side="right", fill="y", padx=(18, 0))
        right.pack_propagate(False)

        self.video = tk.Label(left, bg="#0a1014", text="Camera chưa khởi động", fg="#9db0b8", font=("Segoe UI", 13))
        self.video.pack(fill="both", expand=True, padx=2, pady=2)
        controls = tk.Frame(left, bg="#17252f", pady=12)
        controls.pack(fill="x")
        ttk.Button(controls, text="Bật camera", command=self.start_camera).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Tắt camera", command=self.stop_camera).pack(side="left")
        self.status = tk.Label(controls, text="Sẵn sàng", bg="#17252f", fg="#9db0b8", font=("Segoe UI", 10))
        self.status.pack(side="right", padx=10)

        tk.Label(right, text="Đăng ký khuôn mặt", bg="#f2f0e9", fg="#17252f", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(right, text="Nhập tên rồi bật camera để chụp mẫu.", bg="#f2f0e9", fg="#65727a", wraplength=300, justify="left").pack(anchor="w", pady=(5, 10))
        self.name_entry = ttk.Entry(right)
        self.name_entry.pack(fill="x", pady=(0, 8))
        self.register_button = ttk.Button(right, text="Bắt đầu đăng ký (10 ảnh)", command=self.start_registration)
        self.register_button.pack(fill="x")
        self.progress = tk.Label(right, text="Chưa có phiên đăng ký", bg="#f2f0e9", fg="#65727a")
        self.progress.pack(anchor="w", pady=(8, 20))

        tk.Label(right, text="Người dùng", bg="#f2f0e9", fg="#17252f", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.users_tree = ttk.Treeview(right, columns=("name", "created"), show="headings", height=6)
        self.users_tree.heading("name", text="Tên")
        self.users_tree.heading("created", text="Ngày tạo")
        self.users_tree.column("name", width=145)
        self.users_tree.column("created", width=120)
        self.users_tree.pack(fill="x", pady=(8, 18))
        tk.Label(right, text="Lịch sử gần đây", bg="#f2f0e9", fg="#17252f", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.logs_tree = ttk.Treeview(right, columns=("time", "name", "confidence"), show="headings", height=7)
        for column, title, width in (("time", "Thời gian", 120), ("name", "Tên", 100), ("confidence", "Tin cậy", 70)):
            self.logs_tree.heading(column, text=title)
            self.logs_tree.column(column, width=width)
        self.logs_tree.pack(fill="both", expand=True, pady=(8, 0))

    def start_camera(self):
        if self.running:
            return
        self.status.config(text="Đang kết nối camera...", fg="#ffcf70")
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        for backend in backends:
            camera = cv2.VideoCapture(0, backend)
            if not camera.isOpened():
                camera.release()
                continue
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = camera.read()
            if ok and frame is not None:
                self.camera = camera
                self.pending_frame = frame
                break
            camera.release()
        if self.camera is None:
            messagebox.showerror(
                "Không nhận được hình ảnh",
                "Camera đã mở nhưng không trả về khung hình. Hãy đóng Zoom/Teams/Camera hoặc kiểm tra quyền camera của Windows.",
            )
            self.status.config(text="Camera chưa sẵn sàng", fg="#ff8b7b")
            return
        self.running = True
        self.read_failures = 0
        self.status.config(text="Camera đang chạy", fg="#77e0c2")
        if not self.recognizer.ready:
            self.status.config(text="Camera chạy - chưa có mẫu", fg="#ffcf70")
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.frame_job is not None:
            self.after_cancel(self.frame_job)
            self.frame_job = None
        if self.camera:
            self.camera.release()
            self.camera = None
        self.pending_frame = None
        self.video.config(image="", text="Camera đã tắt")
        self.current_image = None
        self.status.config(text="Camera đã tắt", fg="#9db0b8")

    def start_registration(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Thiếu tên", "Hãy nhập tên người cần đăng ký.")
            return
        if not self.running:
            messagebox.showwarning("Camera chưa bật", "Hãy bật camera trước khi đăng ký.")
            return
        self.register_user_id = self.database.add_user(name)
        (FACES_DIR / str(self.register_user_id)).mkdir(parents=True, exist_ok=True)
        self.register_count = 0
        self.registering = False
        self.register_button.config(state="disabled")
        self.progress.config(text="Đang chụp: 0/10")
        self._capture_registration_samples()

    def _capture_registration_samples(self):
        saved = 0
        for index in range(10):
            if not self.camera:
                break
            ok, frame = self.camera.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)
            boxes = self.recognizer.detect(gray)
            box = self._best_face_box(gray, boxes)
            x, y, width, height = box
            crop = gray[y : y + height, x : x + width]
            path = FACES_DIR / str(self.register_user_id) / f"sample_{saved:02d}.jpg"
            if self.recognizer._write_image(path, crop):
                saved += 1
            self.progress.config(text=f"Đang chụp: {saved}/10")
            self.update_idletasks()
            time.sleep(0.08)
        self.register_button.config(state="normal")
        if saved == 10:
            self.recognizer.train()
            self.refresh_tables()
            self.progress.config(text="Đăng ký hoàn tất")
        else:
            self.progress.config(text=f"Chỉ lưu được {saved}/10 ảnh")

    def _update_frame(self):
        if not self.running or not self.camera:
            return
        self.video.config(text="Đang nhận hình ảnh...")
        if self.pending_frame is not None:
            frame = self.pending_frame
            self.pending_frame = None
            ok = True
        else:
            ok, frame = self.camera.read()
        if not ok:
            self.read_failures += 1
            if self.read_failures >= 5:
                self.stop_camera()
                self.status.config(text="Camera không phản hồi", fg="#ff8b7b")
                messagebox.showwarning("Camera không phản hồi", "Camera không trả về khung hình. Hãy kiểm tra quyền truy cập rồi thử lại.")
                return
            self.status.config(text="Đang chờ khung hình...", fg="#ffcf70")
            self.frame_job = self.after(100, self._update_frame)
            return
        self.read_failures = 0
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.recognizer.detect(gray)
        if len(boxes) > 1:
            boxes = [max(boxes, key=lambda item: item[2] * item[3])]
        if len(boxes) == 0 and (self.registering or self.recognizer.ready):
            boxes = [self._center_box(gray)]
        elif self.recognizer.ready and len(boxes) > 0:
            detected_box = boxes[0]
            center_x = detected_box[0] + detected_box[2] / 2
            frame_center_x = gray.shape[1] / 2
            if abs(center_x - frame_center_x) > gray.shape[1] * 0.3:
                boxes = [self._center_box(gray)]
        if self.registering and len(boxes) == 0:
            self.progress.config(text=f"Đang chụp: {self.register_count}/10 - Đưa mặt vào giữa khung")
        for box in boxes:
            x, y, width, height = box
            name, confidence = self.recognizer.recognize(gray, box)
            if self.registering and self.register_count < 10 and width >= 60 and height >= 60:
                crop = gray[y : y + height, x : x + width]
                path = FACES_DIR / str(self.register_user_id) / f"sample_{self.register_count:02d}.jpg"
                if self.recognizer._write_image(path, crop):
                    self.register_count += 1
                self.progress.config(text=f"Đang chụp: {self.register_count}/10")
                if self.register_count == 10:
                    self.registering = False
                    self.register_button.config(state="normal")
                    self.recognizer.train()
                    self.refresh_tables()
                    self.progress.config(text="Đăng ký hoàn tất")
            label = name or ("Chưa đăng ký" if not self.recognizer.ready else "Người lạ")
            color = (70, 210, 120) if name else (80, 90, 230)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, f"{label}  {confidence:.0f}%", (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            if name and time.time() - self.last_log.get(name, 0) > 10:
                user_id = next((row["id"] for row in self.database.users() if row["name"] == name), None)
                self.database.add_log(user_id, confidence)
                self.last_log[name] = time.time()
                self.refresh_tables()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        max_width, max_height = 760, 620
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        try:
            ok, encoded = cv2.imencode(".ppm", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("Không thể chuyển khung hình sang định dạng hiển thị")
            self.current_image = tk.PhotoImage(data=encoded.tobytes(), format="ppm")
        except Exception as error:
            self.stop_camera()
            messagebox.showerror("Không hiển thị được camera", str(error))
            return
        self.video.image = self.current_image
        self.video.config(image=self.current_image, text="")
        self.video.update_idletasks()
        self.status.config(text="Đang nhận hình ảnh", fg="#77e0c2")
        self.frame_job = self.after(30, self._update_frame)

    @staticmethod
    def _center_box(gray):
        height, width = gray.shape[:2]
        box_width = int(width * 0.5)
        box_height = int(height * 0.65)
        x = (width - box_width) // 2
        y = (height - box_height) // 2
        return x, y, box_width, box_height

    @classmethod
    def _best_face_box(cls, gray, boxes):
        if not len(boxes):
            return cls._center_box(gray)
        frame_center_x = gray.shape[1] / 2
        valid_boxes = [
            box for box in boxes
            if abs(box[0] + box[2] / 2 - frame_center_x) <= gray.shape[1] * 0.3
        ]
        return max(valid_boxes, key=lambda item: item[2] * item[3]) if valid_boxes else cls._center_box(gray)

    def refresh_tables(self):
        for tree in (self.users_tree, self.logs_tree):
            for item in tree.get_children():
                tree.delete(item)
        for row in self.database.users():
            self.users_tree.insert("", "end", values=(row["name"], row["created_at"][:10]))
        for row in self.database.recent_logs(50):
            self.logs_tree.insert("", "end", values=(row["recognized_at"][11:], row["name"], f"{row['confidence']:.0f}%"))

    def close(self):
        self.stop_camera()
        self.database.connection.close()
        self.destroy()


if __name__ == "__main__":
    try:
        FaceApp().mainloop()
    except RuntimeError as error:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Không thể khởi động", str(error))
        root.destroy()