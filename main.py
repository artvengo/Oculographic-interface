import image_processing_package
import cv2
import os

def main():
    print("=" * 60)
    print("ПРОГРАММА ДЕТЕКЦИИ ОБЪЕКТОВ И QR-КОДОВ")
    print("=" * 60)

    detector = image_processing_package.VideoObjDetect(model_type="tiny-yolov3")

    while True:
        print("\n" + "=" * 40)
        print("ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
        print("1. Обработка видеофайла")
        print("2. Детекция с веб-камеры (объекты + QR)")
        print("3. Детекция с веб-камеры (только QR)")
        print("4. Выход")
        print("=" * 40)

        choice = input("Ваш выбор (1, 2, 3 или 4): ").strip()

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

                detector.detect_video(video_file, output_file, fps, min_prob)

                base_output = os.path.splitext(output_file)[0]
                double_file = f"{base_output}.mp4.mp4"
                if os.path.exists(double_file):
                    os.remove(double_file)
                    print(f"Удален лишний файл: {double_file}")

            except ValueError:
                print("Ошибка: введите числа!")

        elif choice == "2":
            try:
                camera_id = input("Введите ID камеры (0 для основной): ").strip()
                camera_id = int(camera_id) if camera_id else 0
                print("\n Инициализация камеры...")
                detector._alternative_camera_detection(camera_id, detect_qr=True)
            except Exception as e:
                print(f" Ошибка: {e}")

        elif choice == "3":
            print("\nЗапуск режима только QR-кодов...")
            try:
                camera_id = input("Введите ID камеры (0 для основной): ").strip()
                camera_id = int(camera_id) if camera_id else 0
                
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print("Не удалось открыть камеру!")
                    continue
                
                print("Камера открыта. Нажмите 'q' для выхода")
                print("Данные QR-кодов будут отображаться на видео и в консоли\n")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    decoded = detector.decode_qr(frame)
                    display_frame = frame.copy()
                    
                    for qr in decoded:
                        rect = qr['rect']
                        x, y, w, h = rect.left, rect.top, rect.width, rect.height
                        
                        # Красная рамка
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        
                        # Данные из QR-кода
                        try:
                            qr_data = qr['data'].decode('utf-8')
                        except:
                            qr_data = str(qr['data'])
                        
                        if len(qr_data) > 40:
                            display_text = qr_data[:37] + "..."
                        else:
                            display_text = qr_data
                        
                        # Рисуем фон под текст
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (text_w, text_h), _ = cv2.getTextSize(display_text, font, 0.6, 2)
                        cv2.rectangle(display_frame,
                                     (x, y - text_h - 8),
                                     (x + text_w + 8, y),
                                     (0, 0, 0), -1)
                        
                        # Зеленый текст
                        cv2.putText(display_frame, display_text,
                                   (x + 4, y - 5),
                                   font, 0.6, (0, 255, 0), 2)
                        
                        # Вывод в консоль
                        print(f"QR-код: {qr_data}")
                    
                    # Статистика
                    cv2.putText(display_frame, f"QR-codes found: {len(decoded)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'q' to quit | 's' screenshot", 
                               (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    cv2.imshow('QR Scanner', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"qr_scan_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"Скриншот сохранен: {filename}")
                
                cap.release()
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == "4":
            print("Программа завершена")
            break
        else:
            print("Неверный выбор!")


if __name__ == "__main__":
    main()