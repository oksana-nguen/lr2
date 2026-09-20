import cv2
import numpy as np
# ядро, которое ходит по изображению и является для него как трафарет
kernel = np.ones((3,3), np.uint8)
#7*7 может искажать объекты и убирать маленькие шумы
#5*5 средняя обработка
#3*3 слабая обработка
cap = cv2.VideoCapture(0) #Подключение к камере
#Границы красного цвета
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
while True:
    # Чтение кадра
    ret, frame_original = cap.read()
    if not ret:
        print("Ошибка захвата кадра!")
        break

    frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2HSV)  # Однократное преобразование в HSV

    mask1 = cv2.inRange(frame, lower_red1, upper_red1) # находим пиксели одной из границ
    mask2 = cv2.inRange(frame,lower_red2, upper_red2) # находим пиксели одной из границ
    red_mask = mask1 +mask2 # накладываем две маски друг на друга
    # 1. ОТКРЫТИЕ - убираем шум, сначала эрозия, потом дилатация
    opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    # 2. ЗАКРЫТИЕ - заполняем дыры, сначала дилатация, потом эрозия
    closing = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    if contours:
        # Берем самый большой контур (или обрабатываем все)
        largest_contour = max(contours, key=cv2.contourArea)

        # Площадь через моменты
        M = cv2.moments(largest_contour)
        area_from_moments = M['m00']
        center_x=M['m10']/M['m00']
        center_y=M['m01']/M['m00']
        cv2.rectangle(frame_original, (int(center_x) - 100, int(center_y) - 100), (int(center_x) + 100, int(center_y) + 100), (0,0,0), 2)
        # m00 - это площадь
        print(area_from_moments)
    #inverted_mask = cv2.bitwise_not(red_mask) # инвертируем объекты из белого в черный, а фон из черного в белый
    cv2.imshow('Camera', red_mask) # выводит только красные пиксели
    cv2.imshow('original', frame_original)
    #cv2.imshow('inverted', inverted_mask)

    # Выход по клавише ESC
    if cv2.waitKey(1) & 0xFF == 27: # выход по кнопке esc
        break