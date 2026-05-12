import cv2
import numpy as np
from multiprocessing import Process, Queue, Event, freeze_support
from queue import Empty, Full
import time
import warnings
import os

try:
    from pyzbar.pyzbar import decode, ZBarSymbol
    QR_SYMBOLS = [ZBarSymbol.QRCODE]
except Exception:
    from pyzbar.pyzbar import decode
    QR_SYMBOLS = None

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"


# ===================== НАСТРОЙКИ =====================
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
CAMERA_FPS = 30
DISPLAY_SCALE = 2
QR_SCAN_EVERY_N_FRAMES = 2

# Расстояние между центрами объективов двух камер в метрах.
# Измерь линейкой максимально точно. Например, 7 см = 0.07.
BASELINE_M = 0.10

# Фокусное расстояние в пикселях.
# Лучше брать из калибровки. Если калибровки нет, значение ниже примерное.
# Для 320 px ширины и угла обзора около 60 градусов получается около 277 px.
FOCAL_LENGTH_PX = 277.0

# Минимальная разница X-координат QR в двух камерах.
# Если disparity слишком маленький, расстояние будет нестабильным.
MIN_DISPARITY_PX = 2.0

# Максимальный допустимый разлёт QR по вертикали после выравнивания камер.
# Если камеры не откалиброваны, можно увеличить до 30-40.
MAX_VERTICAL_DIFF_PX = 25.0

# Опционально: файл калибровки стереопары.
# Если файла нет, программа будет работать без ректификации, но расстояние будет примерным.
CALIBRATION_FILE = "stereo_calibration.npz"
# =====================================================


CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class QRDisplay:
    """Хранит найденные QR-коды и оставляет рамку на экране на несколько секунд."""

    def __init__(self, display_duration=2.0):
        self.qr_codes = {}
        self.display_duration = display_duration

    def update(self, qr_data, points, camera_id):
        points = np.asarray(points, dtype=np.int32)
        center = np.mean(points, axis=0)
        x, y, w, h = cv2.boundingRect(points)

        self.qr_codes[qr_data] = {
            "points": points,
            "center": (float(center[0]), float(center[1])),
            "bbox": (int(x), int(y), int(w), int(h)),
            "timestamp": time.time(),
            "camera": camera_id,
        }

    def get_active_codes(self):
        current_time = time.time()
        active = {}
        expired = []

        for data, info in self.qr_codes.items():
            if current_time - info["timestamp"] <= self.display_duration:
                active[data] = info
            else:
                expired.append(data)

        for data in expired:
            del self.qr_codes[data]

        return active

    def clear(self):
        self.qr_codes.clear()


class StereoRectifier:
    """
    Опциональная ректификация стереокамер.
    Если есть файл calibration.npz, кадры выравниваются перед поиском QR.
    """

    def __init__(self, calibration_file, image_size):
        self.enabled = False
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None
        self.focal_length_px = None
        self.baseline_m = None

        if not calibration_file or not os.path.exists(calibration_file):
            return

        try:
            data = np.load(calibration_file)
            keys = set(data.files)

            if {"map1x", "map1y", "map2x", "map2y"}.issubset(keys):
                self.map1x = data["map1x"]
                self.map1y = data["map1y"]
                self.map2x = data["map2x"]
                self.map2y = data["map2y"]
                self.enabled = True

            elif {"cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "R", "T"}.issubset(keys):
                cm1 = data["cameraMatrix1"]
                dc1 = data["distCoeffs1"]
                cm2 = data["cameraMatrix2"]
                dc2 = data["distCoeffs2"]
                r = data["R"]
                t = data["T"]

                r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
                    cm1, dc1, cm2, dc2, image_size, r, t, alpha=0
                )

                self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                    cm1, dc1, r1, p1, image_size, cv2.CV_32FC1
                )
                self.map2x, self.map2y = cv2.initUndistortRectifyMap(
                    cm2, dc2, r2, p2, image_size, cv2.CV_32FC1
                )

                self.focal_length_px = float(p1[0, 0])
                self.baseline_m = float(np.linalg.norm(t))
                self.enabled = True

        except Exception as exc:
            print(f"Калибровка не загружена: {exc}", flush=True)
            self.enabled = False

    def rectify_pair(self, frame_left, frame_right):
        if not self.enabled:
            return frame_left, frame_right

        left = cv2.remap(frame_left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right = cv2.remap(frame_right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return left, right


def open_camera(camera_id):
    """Открывает камеру через доступные backend'ы."""
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_ANY, "ANY"),
    ]

    for backend, name in backends:
        cap = None
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if not cap.isOpened():
                if cap:
                    cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            time.sleep(0.3)
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0 and np.mean(test_frame) > 5:
                return cap, backend, name

            cap.release()

        except Exception:
            if cap:
                cap.release()

    return None, None, None


def camera_process(camera_id, frame_queue, stop_event, ready_event):
    """Процесс для захвата видео с камеры."""
    cap = None
    used_backend = None

    try:
        cap, used_backend, backend_name = open_camera(camera_id)
        if cap is None:
            return

        ready_event.set()

        error_count = 0
        max_errors = 30

        while not stop_event.is_set():
            try:
                if error_count >= max_errors:
                    cap.release()
                    time.sleep(0.3)
                    cap = cv2.VideoCapture(camera_id, used_backend)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    error_count = 0
                    time.sleep(0.3)
                    continue

                ret, frame = cap.read()

                if not ret or frame is None:
                    time.sleep(0.01)
                    error_count += 1
                    continue

                if frame.size == 0 or np.mean(frame) <= 5:
                    error_count += 1
                    time.sleep(0.033)
                    continue

                if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                error_count = 0

                try:
                    frame_queue.put_nowait(frame)
                except Full:
                    try:
                        frame_queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        frame_queue.put_nowait(frame)
                    except Full:
                        pass

            except Exception:
                error_count += 1
                time.sleep(0.1)

    finally:
        if cap is not None:
            cap.release()


def points_from_pyzbar_object(obj):
    """Преобразует polygon из pyzbar в массив точек OpenCV."""
    raw_points = np.array([(point.x, point.y) for point in obj.polygon], dtype=np.float32)

    if len(raw_points) < 4:
        return None

    if len(raw_points) > 4:
        hull = cv2.convexHull(raw_points)
        return hull.reshape(-1, 2).astype(np.int32)

    return raw_points.astype(np.int32)


def process_qr_code(frame, camera_id, qr_display):
    """Распознавание QR-кодов на кадре."""
    qr_texts = []

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if QR_SYMBOLS is not None:
            decoded_objects = decode(gray, symbols=QR_SYMBOLS)
        else:
            decoded_objects = decode(gray)

        if not decoded_objects:
            enhanced = CLAHE.apply(gray)
            if QR_SYMBOLS is not None:
                decoded_objects = decode(enhanced, symbols=QR_SYMBOLS)
            else:
                decoded_objects = decode(enhanced)

        for obj in decoded_objects:
            try:
                qr_data = obj.data.decode("utf-8", errors="replace")
                points = points_from_pyzbar_object(obj)

                if points is None:
                    continue

                qr_display.update(qr_data, points, camera_id)
                qr_texts.append(qr_data)

            except Exception:
                continue

    except Exception:
        pass

    return frame, qr_texts


def compute_stereo_distances(active_left, active_right, baseline_m, focal_length_px):
    """
    Считает расстояние до QR по формуле Z = f * B / disparity.
    active_left и active_right — активные QR-коды с двух камер.
    """
    distances = {}
    common_codes = set(active_left.keys()) & set(active_right.keys())

    for data in common_codes:
        left_center = active_left[data]["center"]
        right_center = active_right[data]["center"]

        disparity = abs(left_center[0] - right_center[0])
        vertical_diff = abs(left_center[1] - right_center[1])

        if disparity < MIN_DISPARITY_PX:
            continue

        distance_m = (focal_length_px * baseline_m) / disparity

        distances[data] = {
            "distance_m": float(distance_m),
            "disparity_px": float(disparity),
            "vertical_diff_px": float(vertical_diff),
            "low_confidence": vertical_diff > MAX_VERTICAL_DIFF_PX,
        }

    return distances


def draw_qr_codes(frame, active_codes, distance_info=None):
    """Рисует активные QR-коды и, если есть, расстояние до них."""
    if distance_info is None:
        distance_info = {}

    for data, info in active_codes.items():
        points = np.asarray(info["points"], dtype=np.int32)

        cv2.polylines(frame, [points], True, (0, 255, 0), 2)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        x = int(points[0][0])
        y = int(points[0][1]) - 10
        if y < 20:
            y = int(points[3][1]) + 30

        qr_text = data if len(data) < 28 else data[:25] + "..."

        label_lines = [qr_text]
        if data in distance_info:
            dist = distance_info[data]["distance_m"]
            disparity = distance_info[data]["disparity_px"]
            mark = " ?" if distance_info[data]["low_confidence"] else ""
            label_lines.append(f"Z: {dist:.2f} m | d: {disparity:.1f}px{mark}")

        line_height = 16
        max_width = 0
        for line in label_lines:
            (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_width = max(max_width, text_w)

        box_top = y - line_height * len(label_lines) - 4
        box_bottom = y + 4
        cv2.rectangle(frame, (x, box_top), (x + max_width + 6, box_bottom), (0, 0, 0), -1)

        for index, line in enumerate(label_lines):
            text_y = y - line_height * (len(label_lines) - index - 1)
            cv2.putText(frame, line, (x + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return frame


def make_no_signal_frame():
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, "No signal", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def find_available_cameras(max_camera_index=5):
    """Ищет рабочие камеры."""
    available_cameras = []

    for camera_id in range(max_camera_index):
        cap, _, backend_name = open_camera(camera_id)
        if cap is not None:
            available_cameras.append(camera_id)
            print(f"  Камера {camera_id} работает ({backend_name})", flush=True)
            cap.release()

    return available_cameras


def main():
    print("Поиск камер...", flush=True)

    available_cameras = find_available_cameras(max_camera_index=5)
    print(f"Найдено камер: {available_cameras}", flush=True)

    if len(available_cameras) < 2:
        print("Нужно минимум 2 камеры!", flush=True)
        return

    cam1_id = available_cameras[0]
    cam2_id = available_cameras[1]

    print(f"Запуск камер {cam1_id} и {cam2_id}...", flush=True)

    queue1 = Queue(maxsize=1)
    queue2 = Queue(maxsize=1)
    stop_event1 = Event()
    stop_event2 = Event()
    ready_event1 = Event()
    ready_event2 = Event()

    p1 = Process(target=camera_process, args=(cam1_id, queue1, stop_event1, ready_event1))
    p2 = Process(target=camera_process, args=(cam2_id, queue2, stop_event2, ready_event2))

    p1.start()
    p2.start()

    ready1 = ready_event1.wait(timeout=10)
    ready2 = ready_event2.wait(timeout=10)

    print(f"Камера {cam1_id}: {'готова' if ready1 else 'ошибка'}", flush=True)
    print(f"Камера {cam2_id}: {'готова' if ready2 else 'ошибка'}", flush=True)

    if not ready1 and not ready2:
        print("Камеры не готовы!", flush=True)
        return

    qr_display1 = QRDisplay(display_duration=2.0)
    qr_display2 = QRDisplay(display_duration=2.0)

    rectifier = StereoRectifier(CALIBRATION_FILE, (FRAME_WIDTH, FRAME_HEIGHT))

    focal_length_px = FOCAL_LENGTH_PX
    baseline_m = BASELINE_M

    if rectifier.enabled:
        print("Стереокалибровка загружена, кадры будут ректифицированы.", flush=True)
        if rectifier.focal_length_px is not None:
            focal_length_px = rectifier.focal_length_px
        if rectifier.baseline_m is not None and rectifier.baseline_m > 0:
            baseline_m = rectifier.baseline_m
    else:
        print("Калибровка не найдена. Расстояние будет примерным.", flush=True)

    print(f"Baseline: {baseline_m:.4f} m | Focal: {focal_length_px:.1f} px", flush=True)
    print("\nQR-сканер запущен. 'q' - выход, 'c' - очистить QR-коды\n", flush=True)

    last_good1 = None
    last_good2 = None
    frame_counter = 0
    fps_time = time.time()
    fps1 = 0
    fps2 = 0
    frames1 = 0
    frames2 = 0
    last_printed_qr1 = set()
    last_printed_qr2 = set()
    last_distance_print = {}

    try:
        while True:
            try:
                new_frame1 = queue1.get_nowait()
                if new_frame1 is not None:
                    last_good1 = new_frame1
                    frames1 += 1
            except Empty:
                pass
            except Exception:
                pass

            try:
                new_frame2 = queue2.get_nowait()
                if new_frame2 is not None:
                    last_good2 = new_frame2
                    frames2 += 1
            except Empty:
                pass
            except Exception:
                pass

            frame1 = last_good1 if last_good1 is not None else make_no_signal_frame()
            frame2 = last_good2 if last_good2 is not None else make_no_signal_frame()

            if last_good1 is not None and last_good2 is not None:
                frame1, frame2 = rectifier.rectify_pair(frame1, frame2)

            if frame_counter % QR_SCAN_EVERY_N_FRAMES == 0:
                if ready1 and last_good1 is not None:
                    _, qr_texts1 = process_qr_code(frame1.copy(), cam1_id, qr_display1)

                    for text in qr_texts1:
                        if text not in last_printed_qr1:
                            print(f"[Cam {cam1_id}] {text}", flush=True)
                            last_printed_qr1.add(text)

                if ready2 and last_good2 is not None:
                    _, qr_texts2 = process_qr_code(frame2.copy(), cam2_id, qr_display2)

                    for text in qr_texts2:
                        if text not in last_printed_qr2:
                            print(f"[Cam {cam2_id}] {text}", flush=True)
                            last_printed_qr2.add(text)

            frame_counter += 1

            active_codes1 = qr_display1.get_active_codes()
            active_codes2 = qr_display2.get_active_codes()

            distance_info = compute_stereo_distances(
                active_codes1,
                active_codes2,
                baseline_m=baseline_m,
                focal_length_px=focal_length_px,
            )

            current_time = time.time()
            for data, info in distance_info.items():
                previous = last_distance_print.get(data)
                should_print = False

                if previous is None:
                    should_print = True
                else:
                    old_time, old_distance = previous
                    if current_time - old_time > 1.0 and abs(old_distance - info["distance_m"]) > 0.03:
                        should_print = True

                if should_print:
                    quality = "низкая точность" if info["low_confidence"] else "ok"
                    print(
                        f"[Stereo] {data}: {info['distance_m']:.2f} m "
                        f"(disparity={info['disparity_px']:.1f}px, {quality})",
                        flush=True,
                    )
                    last_distance_print[data] = (current_time, info["distance_m"])

            if last_good1 is not None:
                frame1 = draw_qr_codes(frame1.copy(), active_codes1, distance_info)

            if last_good2 is not None:
                frame2 = draw_qr_codes(frame2.copy(), active_codes2, distance_info)

            if time.time() - fps_time >= 1.0:
                fps1 = frames1
                fps2 = frames2
                frames1 = 0
                frames2 = 0
                fps_time = time.time()

            display_width = FRAME_WIDTH * DISPLAY_SCALE
            display_height = FRAME_HEIGHT * DISPLAY_SCALE
            frame1_display = cv2.resize(frame1, (display_width, display_height))
            frame2_display = cv2.resize(frame2, (display_width, display_height))

            cv2.putText(
                frame1_display,
                f"FPS: {fps1} | QR: {len(active_codes1)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame2_display,
                f"FPS: {fps2} | QR: {len(active_codes2)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            combined = np.hstack((frame1_display, frame2_display))
            cv2.line(combined, (display_width, 0), (display_width, display_height), (150, 150, 150), 2)

            cv2.imshow("QR Scanner + Stereo Distance", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                qr_display1.clear()
                qr_display2.clear()
                last_printed_qr1.clear()
                last_printed_qr2.clear()
                last_distance_print.clear()
                print("QR-коды очищены", flush=True)

    finally:
        print("Завершение...", flush=True)
        stop_event1.set()
        stop_event2.set()

        p1.join(timeout=2)
        p2.join(timeout=2)

        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    freeze_support()
    main()
