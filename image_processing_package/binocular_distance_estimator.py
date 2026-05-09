# stereo_distance_estimator.py
import cv2
import numpy as np
import json
import os
from datetime import datetime


class StereoDistanceEstimator:
    """
    Класс для определения расстояния до QR-кодов с помощью двух камер (бинокулярное зрение)
    """
    
    def __init__(self, calibration_file=None):
        """
        Инициализация бинокулярного оценщика расстояний
        
        Параметры:
        calibration_file: файл с параметрами калибровки стереопары
        """
        # Параметры стереопары
        self.Q = None  # Матрица диспаритет-в-глубину
        self.stereo_map = None  # Карты для ректификации
        self.calibrated = False
        
        # Параметры камер
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None  # Матрица поворота между камерами
        self.T = None  # Вектор переноса между камерами
        
        # Параметры стерео алгоритма
        self.stereo_matcher = None
        self.block_size = 15
        self.num_disparities = 16 * 8  # Должно быть кратно 16
        
        # Загрузка калибровки
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)
        
        # История измерений для сглаживания
        self.distance_history = {}
        self.history_length = 5
        
        # Параметры фильтрации
        self.min_disparity = 0
        self.max_disparity = 128
    
    def setup_stereo_matcher(self, algorithm='bm'):
        """
        Настройка стерео матчера
        
        algorithm: 'bm' (Block Matching) или 'sgbm' (Semi-Global Block Matching)
        """
        if algorithm == 'bm':
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=self.block_size
            )
            self.stereo_matcher.setPreFilterType(1)
            self.stereo_matcher.setPreFilterSize(5)
            self.stereo_matcher.setPreFilterCap(61)
            self.stereo_matcher.setTextureThreshold(10)
            self.stereo_matcher.setUniquenessRatio(15)
            self.stereo_matcher.setSpeckleRange(2)
            self.stereo_matcher.setSpeckleWindowSize(100)
            
        elif algorithm == 'sgbm':
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=8 * 3 * self.block_size ** 2,
                P2=32 * 3 * self.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
    
    def calibrate_stereo(self, left_cam_id=0, right_cam_id=1, chessboard_size=(9,6), square_size=0.025):
        """
        Калибровка стереопары с помощью шахматной доски
        
        Параметры:
        left_cam_id: ID левой камеры
        right_cam_id: ID правой камеры
        chessboard_size: размер шахматной доски (внутренние углы)
        square_size: размер квадрата в метрах
        """
        print("\n=== СТЕРЕО КАЛИБРОВКА ===")
        print(f"Размер шахматной доски: {chessboard_size[0]}x{chessboard_size[1]} углов")
        print(f"Размер квадрата: {square_size*100:.1f} см")
        
        # Захват кадров с камер
        cap_left = cv2.VideoCapture(left_cam_id)
        cap_right = cv2.VideoCapture(right_cam_id)
        
        if not cap_left.isOpened() or not cap_right.isOpened():
            print("❌ Ошибка открытия камер!")
            return False
        
        # Подготовка точек для калибровки
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 3D точки в пространстве
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Массивы для хранения точек
        objpoints = []  # 3D точки в реальном мире
        imgpoints_left = []  # 2D точки на левом изображении
        imgpoints_right = []  # 2D точки на правом изображении
        
        print("\nНажмите SPACE для захвата кадра с шахматной доской")
        print("Нажмите ESC для завершения калибровки")
        
        captured_pairs = 0
        min_pairs = 10
        
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not ret_left or not ret_right:
                continue
            
            # Конвертация в оттенки серого
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            
            # Поиск шахматной доски
            ret_left_corners, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
            ret_right_corners, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
            
            # Отображение
            display = np.hstack((frame_left, frame_right))
            
            if ret_left_corners and ret_right_corners:
                cv2.putText(display, "Chessboard detected! Press SPACE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Рисуем углы для отображения
                cv2.drawChessboardCorners(display[:, :frame_left.shape[1]], 
                                         chessboard_size, corners_left, ret_left_corners)
                cv2.drawChessboardCorners(display[:, frame_left.shape[1]:], 
                                         chessboard_size, corners_right, ret_right_corners)
            else:
                cv2.putText(display, "Move chessboard to see both cameras", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(display, f"Captured pairs: {captured_pairs}/{min_pairs}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Stereo Calibration", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and ret_left_corners and ret_right_corners:
                # Уточнение углов
                corners_left_sub = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
                corners_right_sub = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
                
                objpoints.append(objp)
                imgpoints_left.append(corners_left_sub)
                imgpoints_right.append(corners_right_sub)
                
                captured_pairs += 1
                print(f"✅ Захвачена пара {captured_pairs}")
                
            elif key == 27:  # ESC
                break
        
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        
        if captured_pairs < min_pairs:
            print(f"❌ Недостаточно пар для калибровки. Нужно минимум {min_pairs}, получено {captured_pairs}")
            return False
        
        print("\n🔧 Выполнение стерео калибровки...")
        
        # Калибровка левой камеры
        ret, self.camera_matrix_left, self.dist_coeffs_left, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints_left, gray_left.shape[::-1], None, None
        )
        
        # Калибровка правой камеры
        ret, self.camera_matrix_right, self.dist_coeffs_right, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints_right, gray_right.shape[::-1], None, None
        )
        
        # Стерео калибровка
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        
        ret, self.camera_matrix_left, self.dist_coeffs_left, \
        self.camera_matrix_right, self.dist_coeffs_right, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            gray_left.shape[::-1],
            criteria=criteria_stereo,
            flags=flags
        )
        
        # Вычисление матриц ректификации
        rect_left, rect_right, proj_left, proj_right, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            gray_left.shape[::-1], self.R, self.T,
            alpha=0.5
        )
        
        # Вычисление карт ректификации
        self.stereo_map = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, rect_left, proj_left,
            gray_left.shape[::-1], cv2.CV_32FC1
        ), cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, rect_right, proj_right,
            gray_right.shape[::-1], cv2.CV_32FC1
        )
        
        self.calibrated = True
        self.setup_stereo_matcher('sgbm')
        
        print("✅ Стерео калибровка завершена!")
        print(f"База (расстояние между камерами): {np.linalg.norm(self.T):.4f} м")
        
        # Сохранение калибровки
        self.save_calibration("stereo_calibration.json")
        
        return True
    
    def save_calibration(self, filename):
        """Сохранение параметров калибровки"""
        calibration_data = {
            'camera_matrix_left': self.camera_matrix_left.tolist() if self.camera_matrix_left is not None else None,
            'camera_matrix_right': self.camera_matrix_right.tolist() if self.camera_matrix_right is not None else None,
            'dist_coeffs_left': self.dist_coeffs_left.tolist() if self.dist_coeffs_left is not None else None,
            'dist_coeffs_right': self.dist_coeffs_right.tolist() if self.dist_coeffs_right is not None else None,
            'R': self.R.tolist() if self.R is not None else None,
            'T': self.T.tolist() if self.T is not None else None,
            'Q': self.Q.tolist() if self.Q is not None else None,
            'calibrated': self.calibrated
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"📁 Калибровка сохранена в {filename}")
    
    def load_calibration(self, filename):
        """Загрузка параметров калибровки"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix_left = np.array(data['camera_matrix_left']) if data['camera_matrix_left'] else None
            self.camera_matrix_right = np.array(data['camera_matrix_right']) if data['camera_matrix_right'] else None
            self.dist_coeffs_left = np.array(data['dist_coeffs_left']) if data['dist_coeffs_left'] else None
            self.dist_coeffs_right = np.array(data['dist_coeffs_right']) if data['dist_coeffs_right'] else None
            self.R = np.array(data['R']) if data['R'] else None
            self.T = np.array(data['T']) if data['T'] else None
            self.Q = np.array(data['Q']) if data['Q'] else None
            self.calibrated = data['calibrated']
            
            if self.calibrated:
                self.setup_stereo_matcher('sgbm')
                print("✅ Стерео калибровка загружена")
                return True
        except Exception as e:
            print(f"❌ Ошибка загрузки калибровки: {e}")
        
        return False
    
    def compute_disparity(self, left_frame, right_frame):
        """
        Вычисление карты диспаритета
        """
        if not self.calibrated or self.stereo_matcher is None:
            return None
        
        # Конвертация в оттенки серого
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Ректификация
        left_rect = cv2.remap(left_gray, self.stereo_map[0][0], self.stereo_map[0][1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, self.stereo_map[1][0], self.stereo_map[1][1], cv2.INTER_LINEAR)
        
        # Вычисление диспаритета
        disparity = self.stereo_matcher.compute(left_rect, right_rect).astype(np.float32) / 16.0
        
        return disparity, left_rect
    
    def get_qr_disparity(self, qr_left, qr_right, disparity_map):
        """
        Получение диспаритета для QR-кода
        """
        # Центр QR-кода на левом изображении
        center_x = qr_left['x'] + qr_left['width'] // 2
        center_y = qr_left['y'] + qr_left['height'] // 2
        
        # Размер области для поиска соответствия
        search_size = 20
        
        # Поиск соответствующей точки на правом изображении
        min_y = max(0, center_y - search_size)
        max_y = min(disparity_map.shape[0], center_y + search_size)
        min_x = max(0, center_x - search_size)
        max_x = min(disparity_map.shape[1], center_x + search_size)
        
        # Получаем диспаритет в области
        roi = disparity_map[min_y:max_y, min_x:max_x]
        
        # Фильтруем некорректные значения
        valid_disparities = roi[(roi > self.min_disparity) & (roi < self.max_disparity)]
        
        if len(valid_disparities) > 0:
            return np.median(valid_disparities)
        
        return None
    
    def estimate_distance_stereo(self, qr_left, qr_right, disparity_map):
        """
        Оценка расстояния с помощью стерео зрения
        """
        if not self.calibrated or self.Q is None:
            return None
        
        disparity = self.get_qr_disparity(qr_left, qr_right, disparity_map)
        
        if disparity is None or disparity < 1:
            return None
        
        # Преобразование диспаритета в глубину
        # depth = (focal_length * baseline) / disparity
        focal_length = self.camera_matrix_left[0, 0]
        baseline = np.linalg.norm(self.T)
        
        distance = (focal_length * baseline) / disparity
        
        return distance
    
    def estimate_distance_stereo_from_frames(self, qr_detection_left, qr_detection_right, 
                                             left_frame, right_frame, qr_size_m=0.05):
        """
        Полная оценка расстояния с использованием стерео пары
        """
        if not self.calibrated:
            return None
        
        # Вычисление диспаритета
        disparity_map, _ = self.compute_disparity(left_frame, right_frame)
        
        if disparity_map is None:
            return None
        
        # Оценка расстояния
        distance = self.estimate_distance_stereo(qr_detection_left, qr_detection_right, disparity_map)
        
        if distance:
            # Сглаживание
            obj_id = f"qr_{qr_detection_left.get('data', 'unknown')[:20]}"
            distance = self.smooth_distance(obj_id, distance)
            
            # Также можно использовать размер QR-кода для валидации
            size_based_dist = self.estimate_distance_from_size(
                qr_detection_left['width'], qr_size_m
            )
            
            # Комбинируем результаты с весами
            if size_based_dist:
                # Стерео более точен на близких расстояниях, размер - на дальних
                weight_stereo = max(0, 1 - distance / 10)  # Уменьшаем вес на дальних
                weight_size = 1 - weight_stereo
                combined_distance = weight_stereo * distance + weight_size * size_based_dist
            else:
                combined_distance = distance
            
            return {
                'distance_m': combined_distance,
                'distance_cm': combined_distance * 100,
                'disparity': float(disparity_map[qr_detection_left['y'] + qr_detection_left['height']//2,
                                               qr_detection_left['x'] + qr_detection_left['width']//2]) if disparity_map is not None else None,
                'method': 'stereo_vision',
                'baseline_m': float(np.linalg.norm(self.T)) if self.T is not None else None
            }
        
        return None
    
    def estimate_distance_from_size(self, pixel_width, real_width_m):
        """
        Оценка расстояния по размеру (как в монокулярном режиме)
        """
        if not self.calibrated or self.camera_matrix_left is None:
            return None
        
        focal_length = self.camera_matrix_left[0, 0]
        distance = (real_width_m * focal_length) / pixel_width
        return distance
    
    def smooth_distance(self, obj_id, new_distance):
        """
        Сглаживание расстояния
        """
        if obj_id not in self.distance_history:
            self.distance_history[obj_id] = []
        
        self.distance_history[obj_id].append(new_distance)
        
        if len(self.distance_history[obj_id]) > self.history_length:
            self.distance_history[obj_id].pop(0)
        
        return np.median(self.distance_history[obj_id])
    
    def draw_stereo_info(self, frame, detection, distance_info):
        """
        Отрисовка информации о стерео измерении
        """
        x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
        
        dist = distance_info['distance_m']
        
        if dist < 0.5:
            color = (0, 0, 255)
        elif dist < 1.0:
            color = (0, 165, 255)
        elif dist < 2.0:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        text = f"QR: {dist:.2f}m (Stereo)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 2)
        
        cv2.rectangle(frame, (x, y - text_h - 6), (x + text_w + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, y - 3), font, 0.5, (255, 255, 255), 1)
        
        # Дополнительная информация
        if distance_info.get('disparity'):
            info_text = f"Disparity: {distance_info['disparity']:.1f}"
            cv2.putText(frame, info_text, (x, y + h + 15), font, 0.4, (255, 255, 0), 1)
        
        return frame


class StereoQRProcessor:
    """
    Обработчик QR-кодов с использованием двух камер
    """
    
    def __init__(self, left_cam_id=0, right_cam_id=1, calibration_file=None):
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.stereo_estimator = StereoDistanceEstimator(calibration_file)
        self.cap_left = None
        self.cap_right = None
        self.is_running = False
        
        # Параметры для кэширования
        self.last_qr_left = None
        self.last_qr_right = None
        self.last_distance = None
    
    def initialize_cameras(self):
        """Инициализация камер"""
        self.cap_left = cv2.VideoCapture(self.left_cam_id)
        self.cap_right = cv2.VideoCapture(self.right_cam_id)
        
        if not self.cap_left.isOpened():
            print(f"❌ Ошибка: не удалось открыть левую камеру (ID: {self.left_cam_id})")
            return False
        
        if not self.cap_right.isOpened():
            print(f"❌ Ошибка: не удалось открыть правую камеру (ID: {self.right_cam_id})")
            return False
        
        return True
    
    def calibrate(self, chessboard_size=(9,6), square_size=0.025):
        """Калибровка стереопары"""
        if not self.initialize_cameras():
            return False
        
        result = self.stereo_estimator.calibrate_stereo(
            self.left_cam_id, self.right_cam_id, 
            chessboard_size, square_size
        )
        
        self.cap_left.release()
        self.cap_right.release()
        
        return result
    
    def find_matching_qr(self, qr_left, qr_right, max_disparity=100):
        """
        Поиск соответствующего QR-кода на правом изображении
        """
        # Простое сопоставление по положению
        for qr in qr_right:
            # QR-код на правом изображении должен быть смещен влево
            x_diff = qr_left['x'] - qr['x']
            if 0 < x_diff < max_disparity:
                y_diff = abs(qr_left['y'] + qr_left['height']//2 - (qr['y'] + qr['height']//2))
                if y_diff < 50:  # Примерное совпадение по вертикали
                    return qr
        
        return None
    
    def process_frame(self, left_frame, right_frame, qr_size_m=0.05):
        """
        Обработка стереопары для определения расстояния до QR
        """
        # Поиск QR-кодов на обоих кадрах
        from pyzbar.pyzbar import decode
        
        qr_left = []
        qr_right = []
        
        # Поиск на левом кадре
        decoded_left = decode(left_frame)
        for obj in decoded_left:
            rect = obj.rect
            qr_left.append({
                'type': 'qr',
                'data': obj.data.decode('utf-8') if hasattr(obj.data, 'decode') else str(obj.data),
                'x': rect.left,
                'y': rect.top,
                'width': rect.width,
                'height': rect.height
            })
        
        # Поиск на правом кадре
        decoded_right = decode(right_frame)
        for obj in decoded_right:
            rect = obj.rect
            qr_right.append({
                'type': 'qr',
                'data': obj.data.decode('utf-8') if hasattr(obj.data, 'decode') else str(obj.data),
                'x': rect.left,
                'y': rect.top,
                'width': rect.width,
                'height': rect.height
            })
        
        results = []
        
        # Сопоставление QR-кодов и вычисление расстояния
        for ql in qr_left:
            matching_qr = self.find_matching_qr(ql, qr_right)
            
            if matching_qr and self.stereo_estimator.calibrated:
                distance = self.stereo_estimator.estimate_distance_stereo_from_frames(
                    ql, matching_qr, left_frame, right_frame, qr_size_m
                )
                
                if distance:
                    ql['distance'] = distance
                    ql['display_text'] = f"QR: {distance['distance_m']:.2f}m (Stereo)"
                    results.append(ql)
            else:
                # Если стерео не работает, используем оценку по размеру
                if self.stereo_estimator.calibrated and self.stereo_estimator.camera_matrix_left is not None:
                    focal_length = self.stereo_estimator.camera_matrix_left[0, 0]
                    distance_m = (qr_size_m * focal_length) / ql['width']
                    
                    ql['distance'] = {
                        'distance_m': distance_m,
                        'distance_cm': distance_m * 100,
                        'method': 'size_based'
                    }
                    ql['display_text'] = f"QR: {distance_m:.2f}m (Size)"
                    results.append(ql)
        
        return results
    
    def run_detection_loop(self, qr_size_m=0.05):
        """
        Запуск основного цикла детекции с двух камер
        """
        if not self.initialize_cameras():
            return
        
        if not self.stereo_estimator.calibrated:
            print("\n⚠️ Камеры не откалиброваны!")
            print("Сначала выполните калибровку с помощью опции 8 в главном меню")
            return
        
        print("\n🎥 Запуск стерео детекции QR-кодов...")
        print("Нажмите 'q' для выхода, 's' для скриншота")
        
        self.is_running = True
        
        while self.is_running:
            ret_left, left_frame = self.cap_left.read()
            ret_right, right_frame = self.cap_right.read()
            
            if not ret_left or not ret_right:
                print("❌ Ошибка чтения с камер")
                break
            
            # Обработка кадров
            qr_detections = self.process_frame(left_frame, right_frame, qr_size_m)
            
            # Отображение результатов
            display = self.draw_results(left_frame, qr_detections)
            
            # Информация о статусе
            info_text = f"Stereo Detection | Calibrated: {'Yes' if self.stereo_estimator.calibrated else 'No'}"
            if self.stereo_estimator.T is not None:
                info_text += f" | Baseline: {np.linalg.norm(self.stereo_estimator.T):.3f}m"
            
            cv2.putText(display, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "QR codes detected: " + str(len(qr_detections)), 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if qr_detections and qr_detections[0].get('distance'):
                d = qr_detections[0]['distance']
                cv2.putText(display, f"Distance: {d['distance_m']:.2f}m ({d['method']})", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Stereo QR Detection', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"stereo_capture_{timestamp}.jpg", display)
                print(f"📸 Скриншот сохранен: stereo_capture_{timestamp}.jpg")
        
        self.stop()
    
    def draw_results(self, frame, detections):
        """Отрисовка результатов детекции"""
        display = frame.copy()
        
        for det in detections:
            x, y, w, h = det['x'], det['y'], det['width'], det['height']
            
            if 'distance' in det:
                dist = det['distance']['distance_m']
                if dist < 0.5:
                    color = (0, 0, 255)
                elif dist < 1.0:
                    color = (0, 165, 255)
                elif dist < 2.0:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            
            text = det.get('display_text', det.get('data', 'QR Code')[:30])
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 2)
            
            cv2.rectangle(display, (x, y - text_h - 6), (x + text_w + 6, y), color, -1)
            cv2.putText(display, text, (x + 3, y - 3), font, 0.5, (255, 255, 255), 1)
        
        return display
    
    def stop(self):
        """Остановка и освобождение ресурсов"""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        cv2.destroyAllWindows()