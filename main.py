import image_processing_package
import cv2
import os
import time


def main():
    print("=" * 60)
    print("ПРОГРАММА ДЕТЕКЦИИ ОБЪЕКТОВ И QR-КОДОВ")
    print("С ОПРЕДЕЛЕНИЕМ РАССТОЯНИЯ ДО QR-КОДОВ")
    print("=" * 60)

    detector = image_processing_package.VideoObjectDetector(model_type="tiny-yolov3")
    
    # Инициализация Estimator'а расстояний для QR-кодов
    distance_estimator = image_processing_package.DistanceEstimator(focal_length_mm=4.0, sensor_width_mm=6.4)
    
    # Флаг, нужно ли определять расстояние
    enable_distance = True  # По умолчанию определение расстояния включено

    while True:
        print("\n" + "=" * 40)
        print("ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
        print("1. Обработка видеофайла (QR + нейросеть)")
        print("2. Режим камеры (QR + нейросеть)")
        print("3. Режим камеры (только QR)")
        print("4. Режим камеры (только нейросеть)")
        print("5. Калибровка камеры для точного определения расстояния")
        print(f"6. Определение расстояния до QR: {'ВКЛЮЧЕНО' if enable_distance else 'ВЫКЛЮЧЕНО'}")
        print("7. Выход")
        print("=" * 40)

        choice = input("Ваш выбор (1-7): ").strip()

        if choice == "1":
            video_file = input("Введите имя видеофайла: ").strip()
            if not os.path.exists(video_file):
                print(f"Файл {video_file} не найден!")
                continue

            output_file = input("Введите имя выходного файла (Enter для auto): ").strip()
            if not output_file:
                base_name = os.path.splitext(video_file)[0]
                output_file = f"{base_name}_detected.mp4"

            if not output_file.endswith('.mp4'):
                output_file += '.mp4'

            try:
                fps = int(input("Введите FPS (Enter для 20): ").strip() or "20")
                min_prob = int(input("Введите мин. уверенность % (Enter для 50): ").strip() or "50")
                
                # Обработка видео с обеими детекциями и определением расстояния до QR
                process_video_with_distance(detector, distance_estimator, video_file, 
                                           output_file, fps, min_prob, enable_distance)
            except ValueError:
                print("Ошибка: введите числа!")

        elif choice == "2":
            run_camera_mode(detector, distance_estimator, enable_distance,
                          detect_qr=True, detect_neural=True)

        elif choice == "3":
            run_camera_mode(detector, distance_estimator, enable_distance,
                          detect_qr=True, detect_neural=False)

        elif choice == "4":
            run_camera_mode(detector, distance_estimator, enable_distance,
                          detect_qr=False, detect_neural=True)

        elif choice == "5":
            calibrate_camera(detector, distance_estimator)

        elif choice == "6":
            enable_distance = not enable_distance
            print(f"\nОпределение расстояния до QR-кодов: {'ВКЛЮЧЕНО' if enable_distance else 'ВЫКЛЮЧЕНО'}")

        elif choice == "7":
            print("Программа завершена")
            break
        else:
            print("Неверный выбор!")


def process_video_with_distance(detector, distance_estimator, input_path, output_path, 
                                fps=20, min_probability=50, enable_distance=True):
    """
    Обработка видеофайла с определением расстояния до QR-кодов
    """
    if not os.path.exists(input_path):
        print(f"Файл {input_path} не найден!")
        return

    print(f"\nОбработка видео: {input_path}")
    print(f"Определение расстояния до QR: {'Вкл' if enable_distance else 'Выкл'}")
    
    if enable_distance and not distance_estimator.calibrated:
        print("\nВНИМАНИЕ: Камера не откалибрована!")
        print("Расстояния будут приблизительными. Для точных результатов выполните калибровку (пункт 5)")
    
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
    
    # Файл для сохранения информации о расстояниях
    info_file = output_path.replace('.mp4', '_distances.txt')
    distance_log = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Детекция каждые 3 кадра
        if frame_count % 3 == 0:
            all_detections = []
            
            # QR детекция
            qr_results = detector.detect_qr(frame)
            
            # Добавляем расстояние до QR-кодов
            if enable_distance and qr_results:
                qr_with_distance = []
                for qr in qr_results:
                    qr_with_dist = distance_estimator.add_distance_to_detection(qr, frame.shape)
                    qr_with_distance.append(qr_with_dist)
                all_detections.extend(qr_with_distance)
            else:
                all_detections.extend(qr_results)
            
            # Нейросеть
            neural_results = detector.detect_neural(frame, min_probability)
            all_detections.extend(neural_results)
            
            if all_detections:
                last_detections = all_detections
                
                # Логируем расстояния до QR-кодов
                if enable_distance:
                    for det in all_detections:
                        if det['type'] == 'qr' and 'distance' in det:
                            distance_log.append({
                                'frame': frame_count,
                                'data': det['data'][:50],
                                'distance_m': det['distance']['distance_m'],
                                'distance_cm': det['distance']['distance_cm']
                            })
        
        # Отрисовка с информацией о расстоянии
        if enable_distance:
            display_frame = draw_detections_with_distance(frame.copy(), last_detections)
        else:
            display_frame = detector.draw_detections(frame.copy(), last_detections)
        
        # Информация на кадре
        info_text = f"Frame: {frame_count}/{total_frames}"
        qr_count = len([d for d in last_detections if d['type'] == 'qr'])
        neural_count = len([d for d in last_detections if d['type'] == 'neural'])
        info_text += f" | QR: {qr_count} | Objects: {neural_count}"
        
        if enable_distance:
            calib_status = "CALIBRATED" if distance_estimator.calibrated else "UNCALIBRATED"
            info_text += f" | Dist: {calib_status}"
        
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
    
    # Сохраняем информацию о расстояниях
    if enable_distance and distance_log:
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ О РАССТОЯНИЯХ ДО QR-КОДОВ\n")
            f.write("=" * 60 + "\n")
            f.write(f"Видеофайл: {input_path}\n")
            f.write(f"Всего кадров: {total_frames}\n")
            f.write(f"Калибровка: {'Выполнена' if distance_estimator.calibrated else 'Не выполнена'}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, log in enumerate(distance_log, 1):
                f.write(f"{i}. Кадр {log['frame']}:\n")
                f.write(f"   Данные QR: {log['data']}\n")
                f.write(f"   Расстояние: {log['distance_m']:.2f} м ({log['distance_cm']:.1f} см)\n")
                f.write("-" * 40 + "\n")
        
        print(f"\nИнформация о расстояниях сохранена в: {info_file}")
    
    # Закрываем все
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nГотово! Видео сохранено: {output_path}")
    print(f"Обработано кадров: {frame_count}")
    print(f"Время обработки: {time.time() - start_time:.1f}с")


def run_camera_mode(detector, distance_estimator, enable_distance, 
                   detect_qr=True, detect_neural=True):
    """
    Запуск режима камеры с определением расстояния до QR-кодов
    """
    try:
        camera_id = input("Введите ID камеры (0 для основной): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        
        # Режим камеры
        mode_text = []
        if detect_qr and detect_neural:
            mode_text.append("QR + нейросеть")
        elif detect_qr:
            mode_text.append("только QR")
        elif detect_neural:
            mode_text.append("только нейросеть")
        
        if enable_distance and detect_qr:
            mode_text.append("+ расстояние до QR")
        
        print(f"\nЗапуск камеры ({', '.join(mode_text)})...")
        print("Нажмите 'q' для выхода, 's' для скриншота")
        if enable_distance and detect_qr:
            print("Нажмите 'c' для быстрой калибровки")
        
        if enable_distance and detect_qr and not distance_estimator.calibrated:
            print("\nВНИМАНИЕ: Камера не откалибрована!")
            print("Расстояния будут приблизительными. Нажмите 'c' для калибровки")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Ошибка открытия камеры!")
            return
        
        frame_count = 0
        last_detections = []
        
        # Настройка FPS для отображения
        fps_display = 0
        fps_counter = 0
        fps_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Расчет FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Собираем все детекции
            all_detections = []
            
            # QR детекция на каждом кадре
            if detect_qr:
                qr_results = detector.detect_qr(frame)
                
                # Добавляем расстояние до QR-кодов если нужно
                if enable_distance and qr_results:
                    qr_with_distance = []
                    for qr in qr_results:
                        qr_with_dist = distance_estimator.add_distance_to_detection(qr, frame.shape)
                        qr_with_distance.append(qr_with_dist)
                    all_detections.extend(qr_with_distance)
                else:
                    all_detections.extend(qr_results)
            
            # Нейросеть каждый 3-й кадр
            if detect_neural:
                if frame_count % 3 == 0:
                    neural_results = detector.detect_neural(frame, min_probability=40)
                    if neural_results:
                        # Сохраняем нейросеть, QR уже обновлены
                        qr_detections = [d for d in all_detections if d['type'] == 'qr']
                        last_detections = qr_detections + neural_results
                    else:
                        last_detections = all_detections
                else:
                    # Обновляем только QR, нейросеть из кэша
                    qr_detections = [d for d in all_detections if d['type'] == 'qr']
                    neural_cached = [d for d in last_detections if d['type'] == 'neural']
                    last_detections = qr_detections + neural_cached
            else:
                last_detections = all_detections
            
            # Отрисовка с информацией о расстоянии
            if enable_distance and detect_qr:
                display_frame = draw_detections_with_distance(frame.copy(), last_detections)
            else:
                display_frame = detector.draw_detections(frame.copy(), last_detections)
            
            # Информация на экране
            qr_count = len([d for d in last_detections if d['type'] == 'qr'])
            neural_count = len([d for d in last_detections if d['type'] == 'neural'])
            
            info_y = 30
            cv2.putText(display_frame, f"FPS: {fps_display}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"QR: {qr_count} | Objects: {neural_count}", 
                       (150, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if enable_distance and detect_qr:
                calib_status = "CALIBRATED" if distance_estimator.calibrated else "NOT CALIBRATED"
                cv2.putText(display_frame, f"Distance: ON | {calib_status}", 
                           (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Показываем ближайший QR-код
                qr_with_dist = [d for d in last_detections if d['type'] == 'qr' and 'distance' in d]
                if qr_with_dist:
                    closest_qr = min(qr_with_dist, key=lambda x: x['distance']['distance_m'])
                    cv2.putText(display_frame, 
                               f"Closest QR: {closest_qr['distance']['distance_m']:.2f}m", 
                               (10, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(display_frame, "Press 'q' quit | 's' screenshot", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Camera Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"\nСкриншот сохранен: {filename}")
                
                # Сохраняем информацию о расстояниях
                if enable_distance and detect_qr:
                    info_file = f"capture_{timestamp}_qr_info.txt"
                    with open(info_file, 'w', encoding='utf-8') as f:
                        f.write("=" * 50 + "\n")
                        f.write("ИНФОРМАЦИЯ О QR-КОДАХ\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Калибровка: {'Выполнена' if distance_estimator.calibrated else 'Не выполнена'}\n")
                        f.write("=" * 50 + "\n\n")
                        
                        qr_detections = [d for d in last_detections if d['type'] == 'qr']
                        if qr_detections:
                            for i, qr in enumerate(qr_detections, 1):
                                f.write(f"QR-код #{i}:\n")
                                f.write(f"  Данные: {qr.get('data', 'N/A')[:100]}\n")
                                if 'distance' in qr:
                                    f.write(f"  Расстояние: {qr['distance']['distance_m']:.2f} м ({qr['distance']['distance_cm']:.1f} см)\n")
                                    f.write(f"  Размер в пикселях: {qr['distance']['pixel_width']}px\n")
                                f.write("-" * 30 + "\n")
                        else:
                            f.write("QR-коды не обнаружены\n")
                    
                    print(f"Информация о QR-кодах сохранена: {info_file}")
                    
            elif key == ord('c') and enable_distance and detect_qr:
                # Быстрая калибровка по QR-коду
                calibrate_with_qr(distance_estimator, frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {e}")


def draw_detections_with_distance(frame, detections):
    """
    Отрисовка детекций с информацией о расстоянии до QR-кодов
    """
    for det in detections:
        x, y, w, h = det['x'], det['y'], det['width'], det['height']
        
        if det['type'] == 'qr' and 'distance' in det:
            # QR-код с известным расстоянием
            dist = det['distance']['distance_m']
            
            # Цвет в зависимости от расстояния
            if dist < 0.5:
                color = (0, 0, 255)  # Красный - очень близко (<50см)
            elif dist < 1.0:
                color = (0, 165, 255)  # Оранжевый - близко (50-100см)
            elif dist < 2.0:
                color = (0, 255, 255)  # Желтый - средне (1-2м)
            elif dist < 5.0:
                color = (0, 255, 0)  # Зеленый - далеко (2-5м)
            else:
                color = (255, 255, 0)  # Голубой - очень далеко (>5м)
            
            # Текст для отображения
            text = f"QR: {dist:.2f}m"
            if 'data' in det:
                short_data = det['data'][:15] + "..." if len(det['data']) > 15 else det['data']
                text = f"{short_data} | {dist:.2f}m"
        elif det['type'] == 'qr':
            # QR-код без расстояния
            color = (0, 0, 255)
            text = det.get('data', 'QR Code')[:30]
        else:
            # Объект нейросети
            color = (255, 0, 0)  # Синий для объектов
            text = f"{det.get('name', 'Object')} ({det.get('probability', 0):.1f}%)"
        
        # Рисуем рамку
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Фон под текст
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 2)
        cv2.rectangle(frame,
                     (x, y - text_h - 6),
                     (x + text_w + 6, y),
                     color, -1)
        
        # Белый текст
        cv2.putText(frame, text,
                   (x + 3, y - 3),
                   font, 0.5, (255, 255, 255), 1)
    
    return frame


def calibrate_camera(detector, distance_estimator):
    """
    Калибровка камеры с использованием QR-кода известного размера
    """
    print("\n" + "=" * 50)
    print("КАЛИБРОВКА КАМЕРЫ ДЛЯ ОПРЕДЕЛЕНИЯ РАССТОЯНИЯ")
    print("=" * 50)
    print("\nСпособы калибровки:")
    print("1. Использовать QR-код известного размера (рекомендуется)")
    print("2. Использовать любой объект известного размера")
    
    method = input("\nВыберите способ (1 или 2): ").strip()
    
    if method == "1":
        calibrate_with_qr_code(distance_estimator)
    else:
        calibrate_with_object(distance_estimator)


def calibrate_with_qr_code(distance_estimator):
    """
    Калибровка по QR-коду известного размера
    """
    print("\n=== КАЛИБРОВКА ПО QR-КОДУ ===")
    print("Для калибровки вам понадобится QR-код известного размера")
    print("Например, распечатайте QR-код размером 5x5 см")
    
    try:
        camera_id = int(input("\nID камеры (0 для основной): ").strip() or "0")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Ошибка открытия камеры!")
            return False
        
        print("\nРазместите QR-код перед камерой на известном расстоянии")
        
        qr_size_cm = float(input("Введите размер стороны QR-кода (в см): ").strip())
        qr_size_m = qr_size_cm / 100.0
        
        distance_m = float(input("Введите расстояние до QR-кода (в метрах): ").strip())
        
        print("\nНажмите SPACE когда QR-код будет в кадре...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Ищем QR-код
            qr_results = []
            from pyzbar.pyzbar import decode
            decoded = decode(frame)
            
            for obj in decoded:
                rect = obj.rect
                qr_results.append({
                    'width': rect.width,
                    'height': rect.height,
                    'data': obj.data.decode('utf-8') if hasattr(obj.data, 'decode') else str(obj.data)
                })
            
            # Отображаем
            display_frame = frame.copy()
            for qr in qr_results:
                # Рисуем рамку вокруг найденного QR
                cv2.rectangle(display_frame, 
                            (qr.get('x', 0), qr.get('y', 0)), 
                            (qr.get('x', 0) + qr['width'], qr.get('y', 0) + qr['height']), 
                            (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Found QR codes: {len(qr_results)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to calibrate, ESC to cancel", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("QR Calibration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if qr_results:
                    # Берем первый найденный QR-код
                    qr_width_px = qr_results[0]['width']
                    
                    # Рассчитываем фокусное расстояние
                    distance_estimator.focal_length_pixels = (qr_width_px * distance_m) / qr_size_m
                    distance_estimator.calibrated = True
                    distance_estimator.default_qr_size = qr_size_m
                    
                    print(f"\n✅ Калибровка успешна!")
                    print(f"Размер QR-кода: {qr_size_cm} см")
                    print(f"Расстояние: {distance_m} м")
                    print(f"Размер в пикселях: {qr_width_px} px")
                    print(f"Фокусное расстояние: {distance_estimator.focal_length_pixels:.2f} px")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("\n❌ QR-код не найден! Попробуйте еще раз")
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка калибровки: {e}")
    
    return False


def calibrate_with_object(distance_estimator):
    """
    Калибровка по любому объекту известного размера
    """
    print("\n=== КАЛИБРОВКА ПО ОБЪЕКТУ ===")
    
    try:
        camera_id = int(input("ID камеры (0 для основной): ").strip() or "0")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Ошибка открытия камеры!")
            return False
        
        print("\nРазместите объект известного размера перед камерой")
        
        object_width_m = float(input("Введите ширину объекта (в метрах): ").strip())
        distance_m = float(input("Введите расстояние до объекта (в метрах): ").strip())
        
        print("\nНажмите SPACE когда будете готовы выделить объект...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.putText(frame, "Press SPACE to select object", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Object Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        # Выбираем объект
        roi = cv2.selectROI("Select object", frame, False)
        cv2.destroyWindow("Select object")
        
        if roi[2] > 0 and roi[3] > 0:
            object_width_px = roi[2]
            
            # Рассчитываем фокусное расстояние
            distance_estimator.focal_length_pixels = (object_width_px * distance_m) / object_width_m
            distance_estimator.calibrated = True
            
            print(f"\n✅ Калибровка успешна!")
            print(f"Ширина объекта: {object_width_m} м")
            print(f"Расстояние: {distance_m} м")
            print(f"Размер в пикселях: {object_width_px} px")
            print(f"Фокусное расстояние: {distance_estimator.focal_length_pixels:.2f} px")
            
            cap.release()
            cv2.destroyAllWindows()
            return True
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка калибровки: {e}")
    
    return False


def calibrate_with_qr(distance_estimator, frame):
    """
    Быстрая калибровка по QR-коду из текущего кадра
    """
    print("\n=== БЫСТРАЯ КАЛИБРОВКА ПО QR-КОДУ ===")
    
    # Ищем QR-коды в кадре
    from pyzbar.pyzbar import decode
    decoded = decode(frame)
    
    if not decoded:
        print("❌ QR-код не найден в текущем кадре!")
        return False
    
    print(f"Найдено QR-кодов: {len(decoded)}")
    
    try:
        qr_size_cm = float(input("Введите размер стороны QR-кода (в см): ").strip())
        qr_size_m = qr_size_cm / 100.0
        distance_m = float(input("Введите расстояние до QR-кода (в метрах): ").strip())
        
        # Берем первый QR-код
        qr_width_px = decoded[0].rect.width
        
        # Рассчитываем фокусное расстояние
        distance_estimator.focal_length_pixels = (qr_width_px * distance_m) / qr_size_m
        distance_estimator.calibrated = True
        distance_estimator.default_qr_size = qr_size_m
        
        print(f"\n✅ Калибровка успешна!")
        print(f"Фокусное расстояние: {distance_estimator.focal_length_pixels:.2f} px")
        return True
        
    except ValueError:
        print("❌ Ошибка ввода!")
        return False


if __name__ == "__main__":
    main()