import cv2
import numpy as np
import math


class DistanceEstimator:
    """
    Класс для оценки расстояния до объектов и QR-кодов
    Использует известные размеры объектов и калибровку камеры
    """
    
    def __init__(self, camera_matrix=None, dist_coeffs=None, focal_length_mm=None, sensor_width_mm=None):
        """
        Инициализация Estimator'а расстояний
        
        Параметры:
        camera_matrix: матрица камеры (3x3) для точной калибровки
        dist_coeffs: коэффициенты дисторсии
        focal_length_mm: фокусное расстояние в мм (для упрощенного расчета)
        sensor_width_mm: ширина сенсора в мм
        """
        # Стандартные размеры объектов в метрах
        self.known_sizes = {
            'person': 0.6,      # ширина плеч ~60см
            'car': 1.8,         # ширина автомобиля ~1.8м
            'bicycle': 0.6,     # ширина велосипеда ~60см
            'motorcycle': 0.8,  # ширина мотоцикла ~80см
            'bus': 2.5,         # ширина автобуса ~2.5м
            'truck': 2.5,       # ширина грузовика ~2.5м
            'train': 3.0,       # ширина поезда ~3м
            'boat': 2.0,        # ширина лодки ~2м
            'default': 0.5      # размер по умолчанию 50см
        }
        
        # Размер QR-кода по умолчанию (в метрах)
        self.default_qr_size = 0.05  # 5см
        
        # Параметры камеры
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.calibrated = True
        else:
            # Используем упрощенный метод
            self.calibrated = False
            self.focal_length_pixels = None
            self.set_focal_length_from_specs(focal_length_mm, sensor_width_mm)
        
        # История измерений для сглаживания
        self.distance_history = {}
        self.history_length = 5
        
    def set_focal_length_from_specs(self, focal_length_mm=4.0, sensor_width_mm=6.4):
        """
        Установка фокусного расстояния на основе спецификаций камеры
        Для типичной веб-камеры: фокус ~4мм, сенсор 1/2.5" (6.4мм ширина)
        """
        if focal_length_mm and sensor_width_mm and not self.calibrated:
            # Сохраняем физические параметры
            self.focal_length_mm = focal_length_mm
            self.sensor_width_mm = sensor_width_mm
            self.focal_length_pixels = None  # Будет установлен после получения размера кадра
    
    def calibrate_from_frame(self, frame, known_distance_m=1.0, known_width_m=0.1):
        """
        Калибровка камеры по известному объекту
        frame: кадр с объектом
        known_distance_m: известное расстояние до объекта в метрах
        known_width_m: известная ширина объекта в метрах
        """
        height, width = frame.shape[:2]
        
        # Запрашиваем пользователя выделить объект
        print(f"Выделите объект известного размера (ширина {known_width_m}м на расстоянии {known_distance_m}м)")
        print("Нажмите и отпустите кнопку мыши для выбора области")
        
        roi = self.select_roi(frame)
        if roi:
            x, y, w, h = roi
            pixel_width = w
            
            # Рассчитываем фокусное расстояние в пикселях
            # f_pixels = (pixel_width * distance) / real_width
            self.focal_length_pixels = (pixel_width * known_distance_m) / known_width_m
            print(f"Калибровка завершена. Фокусное расстояние: {self.focal_length_pixels:.2f} пикселей")
            self.calibrated = True
            return True
        return False
    
    def select_roi(self, frame):
        """Интерактивный выбор области на изображении"""
        roi = cv2.selectROI("Выберите объект для калибровки", frame, False)
        cv2.destroyWindow("Выберите объект для калибровки")
        if roi[2] > 0 and roi[3] > 0:
            return roi
        return None
    
    def estimate_distance_from_width(self, pixel_width, real_width_m, frame_width_px=None):
        """
        Оценка расстояния по ширине объекта
        
        pixel_width: ширина объекта в пикселях
        real_width_m: реальная ширина объекта в метрах
        frame_width_px: ширина кадра в пикселях (если None, используется предыдущее значение)
        """
        if pixel_width <= 0:
            return None
            
        if self.calibrated and self.focal_length_pixels:
            # Используем формулу: distance = (real_width * focal_length) / pixel_width
            distance = (real_width_m * self.focal_length_pixels) / pixel_width
            return distance
        else:
            # Если не откалиброваны, возвращаем относительное расстояние (0-1)
            return 1.0 / (pixel_width + 1) * 10  # Примерное расстояние
    
    def estimate_qr_distance(self, qr_detection, frame_shape):
        """
        Оценка расстояния до QR-кода
        
        qr_detection: словарь с данными QR-кода (x, y, width, height)
        frame_shape: размер кадра (height, width)
        """
        pixel_width = qr_detection.get('width', 0)
        
        # Получаем размер QR-кода из данных (если закодирован)
        real_width = self.default_qr_size
        qr_data = qr_detection.get('data', '')
        
        # Пытаемся извлечь размер из данных QR-кода
        # Формат: "SIZE:0.1" или "WIDTH:10cm"
        if 'SIZE:' in qr_data or 'WIDTH:' in qr_data:
            try:
                if 'SIZE:' in qr_data:
                    size_str = qr_data.split('SIZE:')[1].split()[0]
                else:
                    size_str = qr_data.split('WIDTH:')[1].split()[0]
                
                # Парсим размер
                if 'cm' in size_str:
                    real_width = float(size_str.replace('cm', '')) / 100.0
                elif 'm' in size_str:
                    real_width = float(size_str.replace('m', ''))
                else:
                    real_width = float(size_str)
            except:
                pass
        
        distance = self.estimate_distance_from_width(pixel_width, real_width)
        
        if distance:
            # Сглаживание
            obj_id = f"qr_{qr_data[:20]}"
            distance = self.smooth_distance(obj_id, distance)
            
            return {
                'distance_m': distance,
                'distance_cm': distance * 100,
                'real_width_m': real_width,
                'pixel_width': pixel_width,
                'method': 'qr_size'
            }
        return None
    
    def estimate_object_distance(self, object_detection):
        """
        Оценка расстояния до объекта, обнаруженного нейросетью
        
        object_detection: словарь с данными объекта (name, width, height)
        """
        obj_name = object_detection.get('name', 'default')
        pixel_width = object_detection.get('width', 0)
        
        # Получаем реальный размер объекта
        real_width = self.known_sizes.get(obj_name, self.known_sizes['default'])
        
        distance = self.estimate_distance_from_width(pixel_width, real_width)
        
        if distance:
            # Сглаживание
            distance = self.smooth_distance(obj_name, distance)
            
            return {
                'distance_m': distance,
                'distance_cm': distance * 100,
                'real_width_m': real_width,
                'pixel_width': pixel_width,
                'object_name': obj_name,
                'method': 'known_size'
            }
        return None
    
    def smooth_distance(self, obj_id, new_distance):
        """
        Сглаживание расстояния для уменьшения дрожания
        """
        if obj_id not in self.distance_history:
            self.distance_history[obj_id] = []
        
        history = self.distance_history[obj_id]
        history.append(new_distance)
        
        # Ограничиваем длину истории
        if len(history) > self.history_length:
            history.pop(0)
        
        # Возвращаем медианное значение
        return np.median(history)
    
    def add_distance_to_detection(self, detection, frame_shape):
        """
        Добавляет информацию о расстоянии к детекции
        """
        if detection['type'] == 'qr':
            distance_info = self.estimate_qr_distance(detection, frame_shape)
        elif detection['type'] == 'neural':
            distance_info = self.estimate_object_distance(detection)
        else:
            return detection
        
        if distance_info:
            detection['distance'] = distance_info
            
            # Добавляем расстояние в текст для отображения
            if 'name' in detection:
                detection['display_text'] = f"{detection['name']} ({detection.get('probability', 0):.1f}%) - {distance_info['distance_m']:.2f}м"
            else:
                detection['display_text'] = f"QR: {distance_info['distance_m']:.2f}м"
        else:
            detection['display_text'] = detection.get('data', detection.get('name', 'Object'))[:30]
        
        return detection
    
    def draw_distance_info(self, frame, detections):
        """
        Отрисовка информации о расстоянии на кадре
        """
        for det in detections:
            if 'distance' in det:
                x, y = det['x'], det['y']
                w, h = det['width'], det['height']
                
                # Информация о расстоянии
                dist = det['distance']['distance_m']
                dist_text = f"{dist:.2f}m"
                
                # Цвет в зависимости от расстояния
                if dist < 1.0:
                    color = (0, 0, 255)  # Красный - очень близко
                elif dist < 3.0:
                    color = (0, 255, 255)  # Желтый - близко
                else:
                    color = (0, 255, 0)  # Зеленый - далеко
                
                # Рисуем рамку с цветом расстояния
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Фон для текста
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(dist_text, font, 0.6, 2)
                cv2.rectangle(frame,
                            (x, y - text_h - 20),
                            (x + text_w + 8, y - 12),
                            color, -1)
                
                # Текст расстояния
                cv2.putText(frame, dist_text,
                           (x + 4, y - 15),
                           font, 0.6, (255, 255, 255), 2)
                
                # Название объекта
                if 'display_text' in det:
                    obj_text = det['display_text'].split('-')[0].strip()
                    cv2.putText(frame, obj_text[:30],
                               (x + 4, y + h - 5),
                               font, 0.5, (255, 255, 255), 1)
            else:
                # Стандартная отрисовка
                x, y, w, h = det['x'], det['y'], det['width'], det['height']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        return frame
    
    def calibrate_interactive(self, cap):
        """
        Интерактивная калибровка с использованием камеры
        """
        print("\n=== ИНТЕРАКТИВНАЯ КАЛИБРОВКА КАМЕРЫ ===")
        print("Для калибровки нужен объект известного размера")
        print("Например: лист A4 (ширина 0.21м), монета (0.025м) и т.д.")
        
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            return False
        
        print("\nВведите параметры объекта:")
        try:
            real_width = float(input("Реальная ширина объекта (в метрах, например 0.21 для A4): "))
            distance = float(input("Расстояние до объекта (в метрах, например 1.0): "))
            
            print("\nТеперь выделите объект на изображении мышкой")
            print("Нажмите и отпустите левую кнопку мыши, затем нажмите ENTER")
            
            roi = self.select_roi(frame)
            if roi:
                x, y, w, h = roi
                self.focal_length_pixels = (w * distance) / real_width
                self.calibrated = True
                print(f"\nКалибровка успешна! Фокусное расстояние: {self.focal_length_pixels:.2f} пикселей")
                return True
        except ValueError:
            print("Ошибка ввода")
        
        return False