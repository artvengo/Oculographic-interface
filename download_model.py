import urllib.request
import os

url = "https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/tiny-yolov3.pt"
filename = "tiny-yolov3.pt"

print(f"Скачиваю {filename}...")
urllib.request.urlretrieve(url, filename)
print(f"Готово! Файл сохранен как: {filename}")
print(f"Размер: {os.path.getsize(filename) / (1024*1024):.1f} МБ")

