from imageai.Detection import VideoObjectDetection, ObjectDetection
from pyzbar.pyzbar import decode
import os
import cv2
import time
import tempfile


class VideoObjDetect:
    def __init__(self, model_type="tiny-yolov3"):
        """
        Инициализация детектора model_type: "tiny-yolov3"
        """
        self.execution_path = os.getcwd()

        # Создаем два детектора: для видео и для изображений
        self.video_detector = VideoObjectDetection()
        self.image_detector = ObjectDetection()

        # Настраиваем видео детектор
        if model_type == "tiny-yolov3":
            self.video_detector.setModelTypeAsTinyYOLOv3()
            self.image_detector.setModelTypeAsTinyYOLOv3()
            model_file = "tiny-yolov3.pt"

        model_path = os.path.join(self.execution_path, model_file)
        self.video_detector.setModelPath(model_path)
        self.image_detector.setModelPath(model_path)

        # Загружаем модели
        self.video_detector.loadModel()
        self.image_detector.loadModel()

    def for_frame(self, frame_number, output_array, returned_frame):
        """
        Функция обратного вызова для каждого кадра при обработке видео
        """
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Кадр: {frame_number} | Обнаружено объектов: {len(output_array)}")
        print("-" * 50)

        object_counts = {}

        for detection in output_array:
            name = detection['name']
            probability = detection['percentage_probability']
            box_points = detection['box_points']

            if name in object_counts:
                object_counts[name] += 1
            else:
                object_counts[name] = 1

            print(f"  • {name}: {probability:.1f}%")

            cv2.rectangle(returned_frame,
                          (box_points[0], box_points[1]),
                          (box_points[2], box_points[3]),
                          (0, 0, 255), 2)

            text = f"{name} ({probability:.1f}%)"
            cv2.putText(returned_frame,
                        text,
                        (box_points[0], box_points[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        y_offset = 30
        cv2.putText(returned_frame,
                    f"Всего объектов: {len(output_array)}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        for i, (obj_name, count) in enumerate(object_counts.items()):
            y_offset += 25
            cv2.putText(returned_frame,
                        f"{obj_name}: {count}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        cv2.imshow('Детекция объектов', returned_frame)
        cv2.waitKey(1)

        return returned_frame
    
    def decode_qr(self, frame):
        """
        Декодирование QR-кодов на кадре
        """
        decoded_objects = decode(frame)
        results = []


        
        for obj in decoded_objects:
            data = obj.data
            rect = obj.rect
            polygon = obj.polygon
            size_px = rect.width

            distance = self.calculate_distance(size_px)

            results.append({
                'data': data,
                'rect': rect,
                'polygon': polygon,
                'type': obj.type,
                'quality': obj.quality,
                'size_px': size_px,
                'distance': distance
            })

        return results
    
    def _alternative_camera_detection(self, camera_id=0, detect_qr=True):
        """
        Метод детекции с камеры с поддержкой QR-кодов
        """
        print("Запуск детекции с камеры...")
        print(f"Попытка открыть камеру с ID: {camera_id}")
        print(f"Детекция QR-кодов: {'Включена' if detect_qr else 'Выключена'}")
        print("\nВся информация будет отображаться прямо на видео")
        print("Нажмите 'q' для выхода, 's' для скриншота\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Не удалось открыть камеру!")
            print("Попробуйте:")
            print("1. sudo chmod 666 /dev/video0")
            print("2. Проверьте, не используется ли камера другим приложением")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ret, test_frame = cap.read()
        if not ret:
            print("Камера открыта, но не передает кадры!")
            cap.release()
            return
        
        print(f"Камера успешно открыта!")
        print(f"Разрешение: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        frame_count = 0
        fps = 0
        fps_count = 0
        prev_time = time.time()
        last_detections = []
        
        window_name = 'Детекция объектов и QR-кодов'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        temp_dir = tempfile.gettempdir()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр")
                continue
            
            frame_count += 1
            fps_count += 1
            
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = fps_count
                fps_count = 0
                prev_time = current_time
            
            # Детекция объектов каждый 5-й кадр
            if frame_count % 5 == 1:
                try:
                    temp_image_path = os.path.join(temp_dir, f"temp_frame_{frame_count}.jpg")
                    cv2.imwrite(temp_image_path, frame)
                    
                    detections = self.image_detector.detectObjectsFromImage(
                        input_image=temp_image_path,
                        output_image_path=os.path.join(temp_dir, f"temp_detected_{frame_count}.jpg"),
                        minimum_percentage_probability=40
                    )
                    
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                    temp_detected = os.path.join(temp_dir, f"temp_detected_{frame_count}.jpg")
                    if os.path.exists(temp_detected):
                        os.remove(temp_detected)
                    
                    if detections:
                        last_detections = detections
                    
                except Exception as e:
                    pass
            
            # Детекция QR-кодов
            qr_results = []
            if detect_qr:
                try:
                    qr_results = self.decode_qr(frame)
                except Exception as e:
                    pass
            
            display_frame = frame.copy()
            
            # Рисуем рамки для объектов
            if last_detections:
                for detection in last_detections:
                    name = detection['name']
                    probability = detection['percentage_probability']
                    box_points = detection['box_points']
                    
                    cv2.rectangle(display_frame,
                                (box_points[0], box_points[1]),
                                (box_points[2], box_points[3]),
                                (0, 0, 255), 2)
                    
                    text = f"{name} ({probability:.1f}%)"
                    # Простой способ отрисовки текста
                    cv2.putText(display_frame, text,
                               (box_points[0], box_points[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Рисуем рамки для QR-кодов и выводим текст
            if qr_results:
                for qr in qr_results:
                    rect = qr['rect']
                    x, y, w, h = rect.left, rect.top, rect.width, rect.height
                    
                    # Красная рамка
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Получаем данные из QR-кода
                    try:
                        qr_data = qr['data'].decode('utf-8')
                    except:
                        qr_data = str(qr['data'])
                    
                    # Обрезаем длинные данные
                    if len(qr_data) > 40:
                        display_text = qr_data[:37] + "..."
                    else:
                        display_text = qr_data
                    
                    # Рисуем черный прямоугольник под текст (фон)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
                    
                    # Черный фон
                    cv2.rectangle(display_frame,
                                 (x, y - text_h - 8),
                                 (x + text_w + 8, y),
                                 (0, 0, 0), -1)
                    
                    # Зеленый текст
                    cv2.putText(display_frame, display_text,
                               (x + 4, y - 5),
                               font, font_scale, (0, 255, 0), thickness)
                    
                    # Дополнительно выводим в консоль для отладки
                    print(f"QR-код: {display_text}")
            
            # Информационная панель
            info_y = 30
            cv2.putText(display_frame, f"FPS: {fps}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            obj_count = len(last_detections) if last_detections else 0
            cv2.putText(display_frame, f"Objects: {obj_count}", (150, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            qr_count = len(qr_results)
            cv2.putText(display_frame, f"QR-codes: {qr_count}", (300, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Press 'q' to quit | 's' screenshot",
                        (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nВыход...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Скриншот сохранен: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Камера закрыта")

    def _alternative_video_detection(self, input_path, output_path, fps=20, min_probability=50):
        """
        Альтернативный метод обработки видео с использованием OpenCV
        """
        print("Запуск альтернативного метода обработки видео...")

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Не удалось открыть видеофайл!")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Параметры видео: {width}x{height}, {original_fps:.2f} fps, {total_frames} кадров")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0
        start_time = time.time()

        cv2.namedWindow('Обработка видео', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Обработка видео', 800, 600)

        last_detections = []
        detection_frame_interval = 3

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                if frame_count > 0:
                    estimated_total = elapsed_time / (frame_count / total_frames)
                    remaining = estimated_total - elapsed_time
                    print(
                        f"Прогресс: {progress:.1f}% | Кадр {frame_count}/{total_frames} | Осталось: {remaining:.1f} сек")

            detections = last_detections

            if frame_count % detection_frame_interval == 0:
                try:
                    try:
                        result = self.image_detector.detectObjectsFromImage(
                            input_image=frame,
                            minimum_percentage_probability=min_probability
                        )
                    except TypeError:
                        try:
                            result = self.image_detector.detectObjectsFromImage(
                                input_image=frame,
                                input_type="array",
                                minimum_percentage_probability=min_probability
                            )
                        except TypeError:
                            result = self.image_detector.detectObjectsFromImage(
                                input_image=frame,
                                input_type="array",
                                output_type="array",
                                minimum_percentage_probability=min_probability
                            )

                    if isinstance(result, tuple) and len(result) == 2:
                        _, detections = result
                    else:
                        detections = result

                    if detections:
                        last_detections = detections

                except Exception as e:
                    print(f"Ошибка детекции на кадре {frame_count}: {e}")

            detected_frame = frame.copy()

            if last_detections and isinstance(last_detections, list):
                for detection in last_detections:
                    if isinstance(detection, dict):
                        name = detection.get('name', 'unknown')
                        probability = detection.get('percentage_probability', 0)
                        box_points = detection.get('box_points', [0, 0, 0, 0])

                        cv2.rectangle(detected_frame,
                                    (box_points[0], box_points[1]),
                                    (box_points[2], box_points[3]),
                                    (0, 0, 255), 2)

                        text = f"{name} {probability:.1f}%"
                        font_scale = 0.7
                        font_thickness = 2
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                        cv2.rectangle(detected_frame,
                                    (box_points[0], box_points[1] - text_size[1] - 10),
                                    (box_points[0] + text_size[0] + 10, box_points[1]),
                                    (0, 0, 0), -1)

                        cv2.putText(detected_frame,
                                    text,
                                    (box_points[0] + 5, box_points[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (0, 255, 0), font_thickness)

            info_y = 30
            cv2.putText(detected_frame,
                        f"Frame: {frame_count}/{total_frames}",
                        (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.putText(detected_frame,
                        f"Objects: {len(last_detections) if last_detections else 0}",
                        (200, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            if frame_count % detection_frame_interval == 0:
                status = "Detecting..."
                status_color = (0, 255, 255)
            else:
                status = "Cached"
                status_color = (255, 255, 255)

            cv2.putText(detected_frame,
                        status,
                        (400, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_color, 2)

            cv2.putText(detected_frame,
                        "Press 'q' to stop",
                        (detected_frame.shape[1] - 200, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            out.write(detected_frame)
            processed_count += 1

            cv2.imshow('Обработка видео', detected_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹Обработка прервана пользователем")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        elapsed_time = time.time() - start_time
        print(f"\nОбработка видео завершена!")
        print(f"Обработано кадров: {processed_count}/{frame_count}")
        print(f"Время обработки: {elapsed_time:.1f} сек")
        print(f"Видео сохранено: {output_path}")

        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Размер файла: {size:.2f} MB")

    def detect_video(self, input_path, output_path, fps=20, min_probability=50):
        """
        Детекция объектов в видеофайле с сохранением результата
        """
        print(f"Обработка видео: {input_path}")
        print(f"Сохранение в: {output_path}")

        if not os.path.exists(input_path):
            print(f"Файл {input_path} не найден!")
            return

        try:
            self.video_detector.detectObjectsFromVideo(
                input_file_path=input_path,
                output_file_path=output_path,
                frames_per_second=fps,
                minimum_percentage_probability=min_probability,
                log_progress=True,
                frame_detection_interval=1,
                per_frame_function=self.for_frame,
                save_detected_video=True,
                video_complete_function=self._video_complete
            )
            print("Обработка видео завершена!")

        except Exception as e:
            print(f"Ошибка при обработке видео: {e}")
            print("\nПробуем альтернативный метод обработки видео...")
            self._alternative_video_detection(input_path, output_path, fps, min_probability)

    def _video_complete(self, output_video_path):
        """
        Функция обратного вызова при завершении обработки видео
        """
        print(f"\nВидео успешно обработано и сохранено: {output_video_path}")

        if os.path.exists(output_video_path):
            size = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"Размер файла: {size:.2f} MB")