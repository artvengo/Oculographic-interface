import image_processing_package
import cv2
import os
import time

def main():
    print("=" * 60)
    print("ПРОГРАММА ДЕТЕКЦИИ ОБЪЕКТОВ И QR-КОДОВ")
    print("=" * 60)

    detector = image_processing_package.video_obj_detect(model_type="tiny-yolov3")

    while True:
        print("\n" + "=" * 40)
        print("ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
        print("1. Обработка видеофайла (QR + нейросеть)")
        print("2. Режим камеры (QR + нейросеть)")
        print("3. Режим камеры (только QR)")
        print("4. Режим камеры (только нейросеть)")
        print("5. Выход")
        print("=" * 40)

        choice = input("Ваш выбор (1, 2, 3, 4 или 5): ").strip()

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
                
                # Обработка видео с обеими детекциями
                detector.process_video(video_file, output_file, fps, min_prob, 
                                     detect_qr=True, detect_neural=True)
            except ValueError:
                print("Ошибка: введите числа!")

        elif choice == "2":
            try:
                camera_id = input("Введите ID камеры (0 для основной): ").strip()
                camera_id = int(camera_id) if camera_id else 0
                
                # Режим камеры с обеими детекциями
                print("\nЗапуск камеры (QR + нейросеть)...")
                print("Нажмите 'q' для выхода, 's' для скриншота")
                
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print("Ошибка открытия камеры!")
                    continue
                
                frame_count = 0
                last_detections = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame_count += 1
                    
                    # Собираем все детекции
                    all_detections = []
                    
                    # QR детекция на каждом кадре
                    qr_results = detector.detect_qr(frame)
                    all_detections.extend(qr_results)
                    
                    # Нейросеть каждый 3-й кадр
                    if frame_count % 3 == 0:
                        neural_results = detector.detect_neural(frame, min_probability=40)
                        if neural_results:
                            last_detections = [d for d in all_detections if d['type'] == 'qr'] + neural_results
                        else:
                            last_detections = all_detections
                    else:
                        # Обновляем только QR, нейросеть из кэша
                        qr_only = [d for d in last_detections if d['type'] == 'qr']
                        neural_cached = [d for d in last_detections if d['type'] == 'neural']
                        last_detections = qr_only + neural_cached
                        # Добавляем новые QR
                        if qr_results:
                            last_detections = qr_results + neural_cached
                    
                    # Отрисовка
                    display_frame = detector.draw_detections(frame.copy(), last_detections)
                    
                    # Информация
                    qr_count = len([d for d in last_detections if d['type'] == 'qr'])
                    neural_count = len([d for d in last_detections if d['type'] == 'neural'])
                    cv2.putText(display_frame, f"QR: {qr_count} | Objects: {neural_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'q' to quit | 's' screenshot", 
                               (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    cv2.imshow('Camera Detection', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"capture_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"Скриншот сохранен: {filename}")
                
                cap.release()
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == "3":
            try:
                camera_id = input("Введите ID камеры (0 для основной): ").strip()
                camera_id = int(camera_id) if camera_id else 0
                
                # Режим только QR
                print("\nЗапуск камеры (только QR)...")
                print("Нажмите 'q' для выхода, 's' для скриншота")
                
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print("Ошибка открытия камеры!")
                    continue
                
                last_detections = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Только QR детекция
                    qr_results = detector.detect_qr(frame)
                    if qr_results:
                        last_detections = qr_results
                    elif len(last_detections) > 0:
                        # Очищаем через 30 кадров если нет QR
                        pass
                    
                    # Отрисовка
                    display_frame = detector.draw_detections(frame.copy(), last_detections)
                    
                    # Информация
                    cv2.putText(display_frame, f"QR-codes: {len(last_detections)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'q' to quit | 's' screenshot", 
                               (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    cv2.imshow('QR Scanner', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"qr_capture_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"Скриншот сохранен: {filename}")
                
                cap.release()
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == "4":
            try:
                camera_id = input("Введите ID камеры (0 для основной): ").strip()
                camera_id = int(camera_id) if camera_id else 0
                
                # Режим только нейросеть
                print("\nЗапуск камеры (только нейросеть)...")
                print("Нажмите 'q' для выхода, 's' для скриншота")
                
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print("Ошибка открытия камеры!")
                    continue
                
                frame_count = 0
                last_detections = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame_count += 1
                    
                    # Нейросеть каждый 3-й кадр
                    if frame_count % 3 == 0:
                        neural_results = detector.detect_neural(frame, min_probability=40)
                        if neural_results:
                            last_detections = neural_results
                    
                    # Отрисовка
                    display_frame = detector.draw_detections(frame.copy(), last_detections)
                    
                    # Информация
                    cv2.putText(display_frame, f"Objects: {len(last_detections)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'q' to quit | 's' screenshot", 
                               (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    cv2.imshow('Neural Detection', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"neural_capture_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"Скриншот сохранен: {filename}")
                
                cap.release()
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == "5":
            print("Программа завершена")
            break
        else:
            print("Неверный выбор!")


if __name__ == "__main__":
    main()