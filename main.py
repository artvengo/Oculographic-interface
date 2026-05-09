# main.py
import image_processing_package
import cv2
import os
import time
import numpy as np
from datetime import datetime


def check_available_cameras(max_cameras=4):
    """
    Проверка доступных камер
    Возвращает список ID доступных камер
    """
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def main():
    print("=" * 60)
    print("ПРОГРАММА ДЕТЕКЦИИ ОБЪЕКТОВ И QR-КОДОВ")
    print("С АВТОМАТИЧЕСКИМ ОПРЕДЕЛЕНИЕМ РАССТОЯНИЯ")
    print("=" * 60)
    
    # Проверка доступных камер
    print("\n🔍 Проверка доступных камер...")
    available_cameras = check_available_cameras()
    
    if not available_cameras:
        print("❌ Нет доступных камер! Программа будет завершена.")
        input("\nНажмите Enter для выхода...")
        return
    
    print(f"✅ Найдено камер: {len(available_cameras)}")
    for i, cam_id in enumerate(available_cameras):
        print(f"   Камера {i+1}: ID {cam_id}")
    
    # Определяем режим работы
    use_stereo = len(available_cameras) >= 2
    stereo_initialized = False
    stereo_processor = None
    
    # Инициализация монокулярного детектора и оценщика
    detector = image_processing_package.VideoObjectDetector(model_type="tiny-yolov3")
    mono_estimator = image_processing_package.DistanceEstimator(focal_length_mm=4.0, sensor_width_mm=6.4)
    
    # Если есть две камеры - пробуем инициализировать стерео
    if use_stereo and len(available_cameras) >= 2:
        print("\n🔧 Обнаружено 2 камеры. Пробуем инициализировать стерео режим...")
        stereo_processor, stereo_initialized = init_stereo_processor(
            available_cameras[0], available_cameras[1]
        )
        
        if stereo_initialized:
            print("✅ Стерео режим активен!")
        else:
            print("⚠️ Стерео режим недоступен (требуется калибровка)")
            print("   Будет использован монокулярный режим для обеих камер")
    
    # Главное меню
    while True:
        print("\n" + "=" * 40)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 40)
        
        # Показываем текущий режим
        if stereo_initialized:
            print("🎯 ТЕКУЩИЙ РЕЖИМ: СТЕРЕО (бинокулярный)")
            print("   - QR-коды: определение расстояния с помощью двух камер")
            print("   - Объекты: обнаружение нейросетью (монокулярно)")
        else:
            print("🎯 ТЕКУЩИЙ РЕЖИМ: МОНОКУЛЯРНЫЙ")
            print("   - QR-коды: определение расстояния по размеру")
            print("   - Объекты: обнаружение нейросетью")
        
        print("\nДОСТУПНЫЕ РЕЖИМЫ РАБОТЫ:")
        print("1. Обработка видеофайла (QR + нейросеть)")
        print("2. Режим камеры (QR + нейросеть) - все доступные камеры")
        print("3. Режим камеры (только QR) - все доступные камеры")
        print("4. Режим камеры (только нейросеть)")
        print("5. Калибровка камеры")
        print("6. Выход")
        print("=" * 40)

        choice = input("Ваш выбор (1-6): ").strip()

        if choice == "1":
            # Обработка видеофайла
            process_video_file(detector, mono_estimator)
            
        elif choice == "2":
            # Режим камеры QR + нейросеть
            run_full_camera_mode(detector, mono_estimator, stereo_processor, 
                                stereo_initialized, available_cameras)
            
        elif choice == "3":
            # Режим камеры только QR
            run_qr_only_mode(detector, mono_estimator, stereo_processor, 
                           stereo_initialized, available_cameras)
            
        elif choice == "4":
            # Режим камеры только нейросеть
            run_neural_only_mode(detector, mono_estimator, available_cameras)
            
        elif choice == "5":
            # Калибровка камеры
            calibrate_camera_menu(detector, mono_estimator, stereo_processor, 
                                 stereo_initialized, available_cameras)
            
        elif choice == "6":
            print("\n👋 Программа завершена")
            if stereo_processor:
                stereo_processor.stop()
            break
        else:
            print("❌ Неверный выбор!")


def init_stereo_processor(left_cam_id=0, right_cam_id=1, calibration_file="stereo_calibration.json"):
    """Инициализация стерео процессора"""
    processor = image_processing_package.StereoQRProcessor(
        left_cam_id=left_cam_id,
        right_cam_id=right_cam_id,
        calibration_file=calibration_file if os.path.exists(calibration_file) else None
    )
    
    if processor.stereo_estimator.calibrated:
        return processor, True
    else:
        return processor, False


def process_video_file(detector, mono_estimator):
    """Обработка видеофайла"""
    print("\n" + "=" * 40)
    print("ОБРАБОТКА ВИДЕОФАЙЛА")
    print("=" * 40)
    
    video_file = input("Введите имя видеофайла: ").strip()
    if not os.path.exists(video_file):
        print(f"❌ Файл {video_file} не найден!")
        return

    output_file = input("Введите имя выходного файла (Enter для auto): ").strip()
    if not output_file:
        base_name = os.path.splitext(video_file)[0]
        output_file = f"{base_name}_detected.mp4"

    if not output_file.endswith('.mp4'):
        output_file += '.mp4'

    try:
        fps = int(input("Введите FPS (Enter для 20): ").strip() or "20")
        min_prob = int(input("Введите мин. уверенность % (Enter для 50): ").strip() or "50")
        
        # Обработка видео
        process_video_with_distance(detector, mono_estimator, video_file, 
                                   output_file, fps, min_prob)
    except ValueError:
        print("❌ Ошибка: введите числа!")


def process_video_with_distance(detector, distance_estimator, input_path, output_path, 
                                fps=20, min_probability=50):
    """Обработка видеофайла с определением расстояния до QR-кодов"""
    if not os.path.exists(input_path):
        print(f"Файл {input_path} не найден!")
        return

    print(f"\n🎬 Обработка видео: {input_path}")
    
    if not distance_estimator.calibrated:
        print("⚠️ ВНИМАНИЕ: Камера не откалибрована!")
        print("   Расстояния будут приблизительными.")
    
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
    
    print(f"📊 Всего кадров: {total_frames}")
    
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
            
            # Добавляем расстояние до QR-кодов (всегда включено)
            if qr_results:
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
                for det in all_detections:
                    if det['type'] == 'qr' and 'distance' in det:
                        distance_log.append({
                            'frame': frame_count,
                            'data': det['data'][:50],
                            'distance_m': det['distance']['distance_m'],
                            'distance_cm': det['distance']['distance_cm']
                        })
        
        # Отрисовка
        display_frame = draw_detections_with_distance(frame.copy(), last_detections)
        
        # Информация на кадре
        info_text = f"Frame: {frame_count}/{total_frames}"
        qr_count = len([d for d in last_detections if d['type'] == 'qr'])
        neural_count = len([d for d in last_detections if d['type'] == 'neural'])
        info_text += f" | QR: {qr_count} | Objects: {neural_count}"
        
        calib_status = "CALIBRATED" if distance_estimator.calibrated else "NOT CALIBRATED"
        info_text += f" | Dist: {calib_status}"
        
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Прогресс
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            remaining = (elapsed / frame_count) * (total_frames - frame_count)
            print(f"📈 Прогресс: {progress:.1f}% | Кадр: {frame_count}/{total_frames} | Осталось: {remaining:.1f}с")
        
        # Сохраняем кадр
        out.write(display_frame)
        
        # Показываем
        cv2.imshow('Processing Video', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏸️ Прервано пользователем")
            break
    
    # Сохраняем информацию о расстояниях
    if distance_log:
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
        
        print(f"\n📄 Информация о расстояниях сохранена в: {info_file}")
    
    # Закрываем все
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Готово! Видео сохранено: {output_path}")
    print(f"📊 Обработано кадров: {frame_count}")
    print(f"⏱️ Время обработки: {time.time() - start_time:.1f}с")


def run_full_camera_mode(detector, mono_estimator, stereo_processor, stereo_initialized, available_cameras):
    """Полный режим камеры: QR + нейросеть"""
    print("\n" + "=" * 40)
    print("РЕЖИМ КАМЕРЫ (QR + НЕЙРОСЕТЬ)")
    print("=" * 40)
    
    if stereo_initialized:
        print("🎯 Используется СТЕРЕО режим для QR-кодов")
        print("   Объекты детектируются нейросетью на левой камере")
        
        # Запуск стерео режима с нейросетью
        run_stereo_with_neural(stereo_processor, detector, mono_estimator)
    else:
        print("🎯 Используется МОНОКУЛЯРНЫЙ режим")
        print("   Запуск на основной камере...")
        
        # Выбираем камеру
        cam_id = available_cameras[0]
        
        print(f"📷 Используется камера ID: {cam_id}")
        print("Нажмите 'q' для выхода, 's' для скриншота")
        print("Нажмите 'c' для быстрой калибровки")
        
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("❌ Ошибка открытия камеры!")
            return
        
        frame_count = 0
        last_detections = []
        
        # Настройка FPS
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
            qr_results = detector.detect_qr(frame)
            if qr_results:
                qr_with_distance = []
                for qr in qr_results:
                    qr_with_dist = mono_estimator.add_distance_to_detection(qr, frame.shape)
                    qr_with_distance.append(qr_with_dist)
                all_detections.extend(qr_with_distance)
            
            # Нейросеть каждый 3-й кадр
            if frame_count % 3 == 0:
                neural_results = detector.detect_neural(frame, min_probability=40)
                if neural_results:
                    last_detections = all_detections + neural_results
                else:
                    last_detections = all_detections
            else:
                neural_cached = [d for d in last_detections if d['type'] == 'neural']
                last_detections = all_detections + neural_cached
            
            # Отрисовка
            display_frame = draw_detections_with_distance(frame.copy(), last_detections)
            
            # Информация на экране
            qr_count = len([d for d in last_detections if d['type'] == 'qr'])
            neural_count = len([d for d in last_detections if d['type'] == 'neural'])
            
            info_y = 30
            cv2.putText(display_frame, f"FPS: {fps_display}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"QR: {qr_count} | Objects: {neural_count}", 
                       (150, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            calib_status = "CALIBRATED" if mono_estimator.calibrated else "NOT CALIBRATED"
            cv2.putText(display_frame, f"Distance: {calib_status}", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Ближайший QR
            qr_with_dist = [d for d in last_detections if d['type'] == 'qr' and 'distance' in d]
            if qr_with_dist:
                closest_qr = min(qr_with_dist, key=lambda x: x['distance']['distance_m'])
                cv2.putText(display_frame, 
                           f"Closest QR: {closest_qr['distance']['distance_m']:.2f}m", 
                           (10, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(display_frame, "Press 'q' quit | 's' screenshot | 'c' calibrate", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Camera Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"\n📸 Скриншот сохранен: {filename}")
            elif key == ord('c'):
                calibrate_with_qr(mono_estimator, frame)
        
        cap.release()
        cv2.destroyAllWindows()


def run_qr_only_mode(detector, mono_estimator, stereo_processor, stereo_initialized, available_cameras):
    """Режим только QR-кодов"""
    print("\n" + "=" * 40)
    print("РЕЖИМ КАМЕРЫ (ТОЛЬКО QR)")
    print("=" * 40)
    
    if stereo_initialized:
        print("🎯 Используется СТЕРЕО режим для QR-кодов")
        
        # Запрос размера QR-кода
        qr_size_m = 0.05
        try:
            qr_size_input = input(f"Введите размер QR-кода в см (Enter для {qr_size_m*100:.0f}): ").strip()
            if qr_size_input:
                qr_size_m = float(qr_size_input) / 100.0
        except:
            pass
        
        stereo_processor.qr_size_m = qr_size_m
        stereo_processor.run_detection_loop(qr_size_m)
        
    else:
        print("🎯 Используется МОНОКУЛЯРНЫЙ режим")
        print("   Запуск на всех доступных камерах...")
        
        # Запускаем обработку для всех доступных камер
        run_monocular_qr_on_all_cameras(detector, mono_estimator, available_cameras)


def run_neural_only_mode(detector, mono_estimator, available_cameras):
    """Режим только нейросети"""
    print("\n" + "=" * 40)
    print("РЕЖИМ КАМЕРЫ (ТОЛЬКО НЕЙРОСЕТЬ)")
    print("=" * 40)
    print("🎯 Используется МОНОКУЛЯРНЫЙ режим")
    
    # Выбираем камеру
    cam_id = available_cameras[0]
    
    print(f"📷 Используется камера ID: {cam_id}")
    print("Нажмите 'q' для выхода, 's' для скриншота")
    
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("❌ Ошибка открытия камеры!")
        return
    
    frame_count = 0
    last_detections = []
    
    fps_display = 0
    fps_counter = 0
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Нейросеть каждый 3-й кадр
        if frame_count % 3 == 0:
            neural_results = detector.detect_neural(frame, min_probability=40)
            if neural_results:
                last_detections = neural_results
        
        # Отрисовка
        display_frame = detector.draw_detections(frame.copy(), last_detections)
        
        # Информация
        cv2.putText(display_frame, f"FPS: {fps_display}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Objects: {len(last_detections)}", 
                   (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' quit | 's' screenshot", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow('Neural Network Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"\n📸 Скриншот сохранен: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()


def run_monocular_qr_on_all_cameras(detector, mono_estimator, available_cameras):
    """Запуск монокулярного QR детектора на всех доступных камерах"""
    caps = []
    for cam_id in available_cameras:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            caps.append((cam_id, cap))
            print(f"📷 Камера ID {cam_id} запущена")
    
    if not caps:
        print("❌ Нет доступных камер!")
        return
    
    print("\nНажмите 'q' для выхода, 's' для скриншота")
    print("Нажмите 'c' для быстрой калибровки")
    
    frame_count = 0
    last_detections_per_cam = {cam_id: [] for cam_id, _ in caps}
    
    while True:
        frames_data = []
        
        # Читаем кадры со всех камер
        for cam_id, cap in caps:
            ret, frame = cap.read()
            if ret:
                frames_data.append((cam_id, frame))
        
        if not frames_data:
            continue
        
        frame_count += 1
        
        # Создаем составное изображение
        display_frames = []
        
        for cam_id, frame in frames_data:
            # Детекция QR
            qr_results = detector.detect_qr(frame)
            
            if qr_results:
                qr_with_distance = []
                for qr in qr_results:
                    qr_with_dist = mono_estimator.add_distance_to_detection(qr, frame.shape)
                    qr_with_distance.append(qr_with_dist)
                last_detections_per_cam[cam_id] = qr_with_distance
            else:
                last_detections_per_cam[cam_id] = []
            
            # Отрисовка
            display_frame = draw_detections_with_distance(frame.copy(), last_detections_per_cam[cam_id])
            
            # Добавляем информацию о камере
            cv2.putText(display_frame, f"Camera ID: {cam_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            qr_count = len(last_detections_per_cam[cam_id])
            cv2.putText(display_frame, f"QR codes: {qr_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            display_frames.append(display_frame)
        
        # Объединяем кадры горизонтально
        if len(display_frames) == 1:
            combined = display_frames[0]
        else:
            combined = np.hstack(display_frames[:2])  # Максимум 2 камеры для отображения
        
        # Информация
        calib_status = "CALIBRATED" if mono_estimator.calibrated else "NOT CALIBRATED"
        cv2.putText(combined, f"Distance: {calib_status}", (10, combined.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(combined, "Press 'q' quit | 's' screenshot | 'c' calibrate", 
                   (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow('Multi-Camera QR Detection', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"multi_cam_capture_{timestamp}.jpg"
            cv2.imwrite(filename, combined)
            print(f"\n📸 Скриншот сохранен: {filename}")
        elif key == ord('c'):
            # Калибровка по первому кадру
            if frames_data:
                calibrate_with_qr(mono_estimator, frames_data[0][1])
    
    # Освобождаем камеры
    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()


def run_stereo_with_neural(stereo_processor, detector, mono_estimator):
    """Стерео режим с параллельной нейросетью на левой камере"""
    if not stereo_processor or not stereo_processor.initialize_cameras():
        print("❌ Ошибка инициализации стерео камер!")
        return
    
    print("\n🎥 Запуск стерео QR детекции + нейросеть на левой камере")
    print("Нажмите 'q' для выхода, 's' для скриншота")
    
    # Создаём окно с возможностью изменения размера
    cv2.namedWindow('Stereo + Neural Detection', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    last_neural_detections = []
    
    fps_display = 0
    fps_counter = 0
    fps_time = time.time()
    
    while stereo_processor.is_running:
        ret_left, left_frame = stereo_processor.cap_left.read()
        ret_right, right_frame = stereo_processor.cap_right.read()
        
        if not ret_left or not ret_right:
            print("❌ Ошибка чтения с камер")
            break
        
        frame_count += 1
        
        # FPS счётчик
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Стерео детекция QR (с проверкой на наличие калибровки)
        if stereo_processor.stereo_estimator.calibrated:
            qr_results = stereo_processor.process_frame(left_frame, right_frame)
        else:
            qr_results = []
            cv2.putText(left_frame, "STEREO NOT CALIBRATED!", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Нейросеть на левом кадре (каждый 3-й)
        if frame_count % 3 == 0:
            neural_results = detector.detect_neural(left_frame, min_probability=40)
            if neural_results:
                last_neural_detections = neural_results
        
        # Отрисовка на левом кадре
        display_frame = stereo_processor.draw_results(left_frame.copy(), qr_results)
        
        # Добавляем нейросеть
        for det in last_neural_detections:
            x, y, w, h = det['x'], det['y'], det['width'], det['height']
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = f"{det['name']} ({det['probability']:.1f}%)"
            cv2.putText(display_frame, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Информационная панель
        info_y = 30
        cv2.putText(display_frame, f"FPS: {fps_display}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"QR: {len(qr_results)} | Objects: {len(last_neural_detections)}", 
                   (150, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if stereo_processor.stereo_estimator.T is not None:
            baseline = np.linalg.norm(stereo_processor.stereo_estimator.T)
            cv2.putText(display_frame, f"Baseline: {baseline:.3f}m", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(display_frame, "Press 'q' quit | 's' screenshot", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow('Stereo + Neural Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"stereo_neural_{timestamp}.jpg", display_frame)
            print(f"\n📸 Скриншот сохранен: stereo_neural_{timestamp}.jpg")
    
    stereo_processor.stop()


def calibrate_camera_menu(detector, mono_estimator, stereo_processor, stereo_initialized, available_cameras):
    """Меню калибровки камер"""
    print("\n" + "=" * 40)
    print("КАЛИБРОВКА КАМЕР")
    print("=" * 40)
    
    print("1. Калибровка монокулярной камеры (по QR-коду)")
    print("2. Калибровка монокулярной камеры (по любому объекту)")
    
    if len(available_cameras) >= 2:
        print("3. Калибровка стерео камер (по шахматной доске)")
        print("4. Быстрая калибровка стерео камер (по QR-коду)")
    
    choice = input("\nВаш выбор: ").strip()
    
    if choice == "1":
        calibrate_monocular_with_qr(detector, mono_estimator, available_cameras)
    elif choice == "2":
        calibrate_monocular_with_object(mono_estimator, available_cameras)
    elif choice == "3" and len(available_cameras) >= 2:
        calibrate_stereo_cameras(available_cameras)
    elif choice == "4" and len(available_cameras) >= 2:
        quick_stereo_calibration(available_cameras)
    else:
        print("❌ Неверный выбор!")

def quick_stereo_calibration(available_cameras):
    """
    Быстрая калибровка стерео камер с помощью QR-кода.
    Этот метод упрощает калибровку, но даёт приблизительный результат.
    """
    print("\n=== БЫСТРАЯ КАЛИБРОВКА СТЕРЕО КАМЕР ПО QR-КОДУ ===")
    print("\n📋 Инструкция:")
    print("   1. Разместите QR-код перед камерами на известном расстоянии")
    print("   2. QR-код должен быть виден обеими камерами")
    print("   3. Программа сама рассчитает базовое расстояние\n")
    
    left_id = available_cameras[0]
    right_id = available_cameras[1] if len(available_cameras) > 1 else 1
    
    try:
        qr_size_cm = float(input("Введите размер стороны QR-кода (в см): ").strip())
        qr_size_m = qr_size_cm / 100.0
        distance_to_qr_m = float(input("Введите расстояние до QR-кода (в метрах): ").strip())
    except ValueError:
        print("❌ Ошибка ввода!")
        return
    
    print("\n🎥 Открываю камеры...")
    cap_left = cv2.VideoCapture(left_id)
    cap_right = cv2.VideoCapture(right_id)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("❌ Ошибка открытия камер!")
        return
    
    print("\nНажмите SPACE для калибровки, ESC для отмены")
    
    from pyzbar.pyzbar import decode
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            continue
        
        # Ищем QR-коды
        qr_left = decode(frame_left)
        qr_right = decode(frame_right)
        
        # Отображаем
        display = np.hstack((frame_left, frame_right))
        
        if qr_left and qr_right:
            cv2.putText(display, "QR detected on both cameras! Press SPACE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Рисуем рамки
            for qr in qr_left:
                rect = qr.rect
                cv2.rectangle(display[:, :frame_left.shape[1]], 
                            (rect.left, rect.top), 
                            (rect.left + rect.width, rect.top + rect.height), 
                            (0, 255, 0), 2)
            
            for qr in qr_right:
                rect = qr.rect
                cv2.rectangle(display[:, frame_left.shape[1]:], 
                            (rect.left, rect.top), 
                            (rect.left + rect.width, rect.top + rect.height), 
                            (0, 255, 0), 2)
        else:
            cv2.putText(display, "Move QR code to be visible by both cameras", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(display, f"Left: {len(qr_left)} | Right: {len(qr_right)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Quick Stereo Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if qr_left and qr_right:
                break
            else:
                print("❌ QR-код не виден обеими камерами!")
        elif key == 27:
            cap_left.release()
            cap_right.release()
            cv2.destroyAllWindows()
            return
    
    print("\n🔧 Выполняю расчёт параметров...")
    
    # Получаем центры QR-кодов
    qr_left_center = (qr_left[0].rect.left + qr_left[0].rect.width // 2, 
                      qr_left[0].rect.top + qr_left[0].rect.height // 2)
    qr_right_center = (qr_right[0].rect.left + qr_right[0].rect.width // 2, 
                       qr_right[0].rect.top + qr_right[0].rect.height // 2)
    
    # Размер QR-кода в пикселях (для оценки фокусного расстояния)
    qr_width_px = qr_left[0].rect.width
    
    # Рассчитываем фокусное расстояние (для монокулярной части)
    focal_length_px = (qr_width_px * distance_to_qr_m) / qr_size_m
    
    # Рассчитываем диспаритет (смещение между камерами)
    disparity = abs(qr_left_center[0] - qr_right_center[0])
    
    # Рассчитываем базовое расстояние между камерами
    # baseline = (disparity * distance) / focal_length
    baseline_m = (disparity * distance_to_qr_m) / focal_length_px
    
    print(f"\n📊 Результаты быстрой калибровки:")
    print(f"   Фокусное расстояние: {focal_length_px:.2f} px")
    print(f"   Диспаритет: {disparity:.2f} px")
    print(f"   Базовое расстояние: {baseline_m:.4f} м ({baseline_m*100:.1f} см)")
    
    # Создаем калибровочный файл
    calibration_data = {
        'camera_matrix_left': [[focal_length_px, 0, frame_left.shape[1]/2],
                               [0, focal_length_px, frame_left.shape[0]/2],
                               [0, 0, 1]],
        'camera_matrix_right': [[focal_length_px, 0, frame_right.shape[1]/2],
                                [0, focal_length_px, frame_right.shape[0]/2],
                                [0, 0, 1]],
        'dist_coeffs_left': [[0, 0, 0, 0, 0]],
        'dist_coeffs_right': [[0, 0, 0, 0, 0]],
        'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'T': [[baseline_m], [0], [0]],
        'Q': [[1, 0, 0, -frame_left.shape[1]/2],
              [0, 1, 0, -frame_left.shape[0]/2],
              [0, 0, 0, focal_length_px],
              [0, 0, 1/baseline_m, 0]],
        'calibrated': True
    }
    
    with open('stereo_calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\n✅ Калибровочный файл сохранён: stereo_calibration.json")
    print("   Теперь можно использовать стерео режим!")
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

def calibrate_monocular_with_qr(detector, mono_estimator, available_cameras):
    """Калибровка монокулярной камеры по QR-коду"""
    print("\n=== КАЛИБРОВКА ПО QR-КОДУ ===")
    
    cam_id = available_cameras[0]
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        print("❌ Ошибка открытия камеры!")
        return
    
    print("\nРазместите QR-код перед камерой на известном расстоянии")
    
    try:
        qr_size_cm = float(input("Введите размер стороны QR-кода (в см): ").strip())
        qr_size_m = qr_size_cm / 100.0
        distance_m = float(input("Введите расстояние до QR-кода (в метрах): ").strip())
    except ValueError:
        print("❌ Ошибка ввода!")
        cap.release()
        return
    
    print("\nНажмите SPACE когда QR-код будет в кадре...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Ищем QR-код
        qr_results = detector.detect_qr(frame)
        
        # Отображаем
        display_frame = frame.copy()
        for qr in qr_results:
            x, y, w, h = qr['x'], qr['y'], qr['width'], qr['height']
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Size: {w}px", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(display_frame, f"Found QR codes: {len(qr_results)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to calibrate, ESC to cancel", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("QR Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if qr_results:
                qr_width_px = qr_results[0]['width']
                mono_estimator.focal_length_pixels = (qr_width_px * distance_m) / qr_size_m
                mono_estimator.calibrated = True
                
                print(f"\n✅ Калибровка успешна!")
                print(f"   Фокусное расстояние: {mono_estimator.focal_length_pixels:.2f} px")
                break
            else:
                print("\n❌ QR-код не найден!")
        elif key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


def calibrate_monocular_with_object(mono_estimator, available_cameras):
    """Калибровка монокулярной камеры по любому объекту"""
    print("\n=== КАЛИБРОВКА ПО ОБЪЕКТУ ===")
    
    cam_id = available_cameras[0]
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        print("❌ Ошибка открытия камеры!")
        return
    
    try:
        object_width_m = float(input("Введите ширину объекта (в метрах): ").strip())
        distance_m = float(input("Введите расстояние до объекта (в метрах): ").strip())
    except ValueError:
        print("❌ Ошибка ввода!")
        cap.release()
        return
    
    print("\nНажмите SPACE чтобы выделить объект...")
    
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
            return
    
    roi = cv2.selectROI("Select object", frame, False)
    cv2.destroyWindow("Select object")
    
    if roi[2] > 0 and roi[3] > 0:
        object_width_px = roi[2]
        mono_estimator.focal_length_pixels = (object_width_px * distance_m) / object_width_m
        mono_estimator.calibrated = True
        
        print(f"\n✅ Калибровка успешна!")
        print(f"   Фокусное расстояние: {mono_estimator.focal_length_pixels:.2f} px")
    
    cap.release()
    cv2.destroyAllWindows()


def calibrate_stereo_cameras(available_cameras):
    """Калибровка стерео камер с помощью шахматной доски"""
    print("\n=== КАЛИБРОВКА СТЕРЕО КАМЕР ПО ШАХМАТНОЙ ДОСКЕ ===")
    print("\n📋 Инструкция:")
    print("   1. Распечатайте шахматную доску (шаблон можно найти в интернете)")
    print("   2. Разместите её перед камерами")
    print("   3. Меняйте угол и положение доски в кадре")
    print("   4. Нажмите SPACE для захвата кадра (нужно 10-15 кадров)")
    print("   5. Нажмите ESC для завершения калибровки\n")
    
    left_id = available_cameras[0]
    right_id = available_cameras[1] if len(available_cameras) > 1 else 1
    
    print(f"📷 Левая камера: ID {left_id}")
    print(f"📷 Правая камера: ID {right_id}")
    
    try:
        board_width = int(input("Введите количество углов по ширине (9 для A4): ").strip() or "9")
        board_height = int(input("Введите количество углов по высоте (6 для A4): ").strip() or "6")
        square_cm = float(input("Введите размер квадрата в см (2.5): ").strip() or "2.5")
        square_m = square_cm / 100.0
    except ValueError:
        print("❌ Ошибка ввода!")
        return
    
    # Создаём процессор и запускаем калибровку
    processor = image_processing_package.StereoQRProcessor(left_id, right_id)
    success = processor.calibrate((board_width, board_height), square_m)
    
    if success:
        print("\n✅ Стерео калибровка успешна!")
        print("   Теперь стерео режим будет работать автоматически")
    else:
        print("\n❌ Ошибка калибровки!")
        print("   Убедитесь, что шахматная доска хорошо видна обеими камерами")


def calibrate_with_qr(distance_estimator, frame):
    """Быстрая калибровка по QR-коду из текущего кадра"""
    print("\n=== БЫСТРАЯ КАЛИБРОВКА ПО QR-КОДУ ===")
    
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
        
        qr_width_px = decoded[0].rect.width
        distance_estimator.focal_length_pixels = (qr_width_px * distance_m) / qr_size_m
        distance_estimator.calibrated = True
        distance_estimator.default_qr_size = qr_size_m
        
        print(f"\n✅ Калибровка успешна!")
        print(f"   Фокусное расстояние: {distance_estimator.focal_length_pixels:.2f} px")
        return True
        
    except ValueError:
        print("❌ Ошибка ввода!")
        return False


def draw_detections_with_distance(frame, detections):
    """Отрисовка детекций с информацией о расстоянии"""
    for det in detections:
        x, y, w, h = det['x'], det['y'], det['width'], det['height']
        
        if det['type'] == 'qr' and 'distance' in det:
            dist = det['distance']['distance_m']
            
            if dist < 0.5:
                color = (0, 0, 255)
            elif dist < 1.0:
                color = (0, 165, 255)
            elif dist < 2.0:
                color = (0, 255, 255)
            elif dist < 5.0:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            
            method = det['distance'].get('method', 'unknown')
            method_text = "Stereo" if method == 'stereo_vision' else "Size"
            
            text = f"QR: {dist:.2f}m ({method_text})"
            if 'data' in det:
                short_data = det['data'][:15] + "..." if len(det['data']) > 15 else det['data']
                text = f"{short_data} | {dist:.2f}m"
        elif det['type'] == 'qr':
            color = (0, 0, 255)
            text = det.get('data', 'QR Code')[:30]
        else:
            color = (255, 0, 0)
            text = f"{det.get('name', 'Object')} ({det.get('probability', 0):.1f}%)"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 2)
        cv2.rectangle(frame, (x, y - text_h - 6), (x + text_w + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, y - 3), font, 0.5, (255, 255, 255), 1)
    
    return frame


if __name__ == "__main__":
    main()