from imageai.Detection import ObjectDetection
from pyzbar.pyzbar import decode
import os
import cv2
import time
import tempfile


class video_obj_detect:
    def __init__(self, model_type="tiny-yolov3"):
        """
        Инициализация детектора
        """
        self.execution_path = os.getcwd()

        # Детектор для изображений
        self.image_detector = ObjectDetection()

        # Настройка TinyYOLOv3
        if model_type == "tiny-yolov3":
            self.image_detector.setModelTypeAsTinyYOLOv3()
            model_file = "tiny-yolov3.pt"

        model_path = os.path.join(self.execution_path, model_file)
        self.image_detector.setModelPath(model_path)

        # Загрузка модели
        self.image_detector.loadModel()
        
        # Кэш для детекций
        self.last_qr_detections = []
        self.last_neural_detections = []
        self.last_detection_time = 0

    def detect_qr(self, frame):
        """
        Распознавание QR-кодов на кадре
        Возвращает список с координатами и данными
        """
        decoded_objects = decode(frame)
        results = []

        for obj in decoded_objects:
            try:
                data = obj.data.decode('utf-8')
            except:
                data = str(obj.data)
            
            rect = obj.rect
            
            results.append({
                'type': 'qr',
                'data': data,
                'x': rect.left,
                'y': rect.top,
                'width': rect.width,
                'height': rect.height
            })

        return results

    def detect_neural(self, frame, min_probability=40):
        """
        Распознавание объектов нейросетью
        Возвращает список с координатами и названиями объектов
        """
        try:
            # Временный файл для обработки
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, frame)
            
            # Детекция
            detections = self.image_detector.detectObjectsFromImage(
                input_image=temp_path,
                output_image_path=os.path.join(tempfile.gettempdir(), "temp_detected.jpg"),
                minimum_percentage_probability=min_probability
            )
            
            # Удаление временных файлов
            os.unlink(temp_path)
            temp_detected = os.path.join(tempfile.gettempdir(), "temp_detected.jpg")
            if os.path.exists(temp_detected):
                os.unlink(temp_detected)
            
            # Приводим к единому формату
            results = []
            if detections:
                for detection in detections:
                    box_points = detection.get('box_points', [0, 0, 0, 0])
                    results.append({
                        'type': 'neural',
                        'name': detection.get('name', 'unknown'),
                        'probability': detection.get('percentage_probability', 0),
                        'x': box_points[0],
                        'y': box_points[1],
                        'x2': box_points[2],
                        'y2': box_points[3],
                        'width': box_points[2] - box_points[0],
                        'height': box_points[3] - box_points[1]
                    })
            
            return results
            
        except Exception as e:
            print(f"Ошибка нейросети: {e}")
            return []

    def draw_detections(self, frame, detections):
        """
        Отрисовка детекций на кадре (одинаковые рамки для всех)
        detections: список детекций от detect_qr или detect_neural
        """
        for det in detections:
            if det['type'] == 'qr':
                # QR-код
                x, y, w, h = det['x'], det['y'], det['width'], det['height']
                text = det['data']
                if len(text) > 40:
                    text = text[:37] + "..."
            else:
                # Нейросеть
                x, y, w, h = det['x'], det['y'], det['width'], det['height']
                text = f"{det['name']} ({det['probability']:.1f}%)"
            
            # Одинаковая красная рамка для всех
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Фон под текст
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(text, font, 0.6, 2)
            cv2.rectangle(frame,
                         (x, y - text_h - 8),
                         (x + text_w + 8, y),
                         (0, 0, 0), -1)
            
            # Белый текст
            cv2.putText(frame, text,
                       (x + 4, y - 5),
                       font, 0.6, (255, 255, 255), 2)
        
        return frame

    def process_video(self, input_path, output_path, fps=20, min_probability=50, detect_qr=True, detect_neural=True):
        """
        Обработка видеофайла с возможностью выбора типов детекции
        """
        if not os.path.exists(input_path):
            print(f"Файл {input_path} не найден!")
            return

        print(f"Обработка видео: {input_path}")
        print(f"QR детекция: {'Вкл' if detect_qr else 'Выкл'}")
        print(f"Нейросеть: {'Вкл' if detect_neural else 'Выкл'}")
        
        # Открываем видео
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Создаем writer для сохранения
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Всего кадров: {total_frames}")
        
        # Кэш для детекций
        last_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Детекция каждые 3 кадра
            if frame_count % 3 == 0:
                all_detections = []
                
                # QR детекция
                if detect_qr:
                    qr_results = self.detect_qr(frame)
                    all_detections.extend(qr_results)
                
                # Нейросеть
                if detect_neural:
                    neural_results = self.detect_neural(frame, min_probability)
                    all_detections.extend(neural_results)
                
                if all_detections:
                    last_detections = all_detections
            
            # Отрисовка
            display_frame = self.draw_detections(frame.copy(), last_detections)
            
            # Информация
            info_text = f"Frame: {frame_count}/{total_frames}"
            if detect_qr:
                qr_count = len([d for d in last_detections if d['type'] == 'qr'])
                info_text += f" | QR: {qr_count}"
            if detect_neural:
                neural_count = len([d for d in last_detections if d['type'] == 'neural'])
                info_text += f" | Objects: {neural_count}"
            
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Прогресс
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                remaining = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"Прогресс: {progress:.1f}% | Кадр: {frame_count}/{total_frames} | Осталось: {remaining:.1f}с")
            
            # Сохраняем кадр
            out.write(display_frame)
            
            # Показываем
            cv2.imshow('Processing Video', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Прервано пользователем")
                break
        
        # Закрываем все
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nГотово! Видео сохранено: {output_path}")
        print(f"Обработано кадров: {frame_count}")
        print(f"Время обработки: {time.time() - start_time:.1f}с")