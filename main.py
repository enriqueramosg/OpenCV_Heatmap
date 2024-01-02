# cSpell:disable

#-------LIBRERÍAS-------------------------------------------------

import cv2
import numpy as np
from typing import Tuple
import tkinter as tk
from tkinter import filedialog


#-------FUNCIONES-------------------------------------------------

def apply_morph(image: np.ndarray,
                morph_type=cv2.MORPH_CLOSE,
                kernel_size: Tuple[int, int] = (3, 3),
                make_gaussian: bool = True):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if make_gaussian:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.morphologyEx(image, morph_type, kernel)

def add_images(image1: np.ndarray, 
                image2: np.ndarray) -> np.ndarray:
    return np.array(image1, dtype=np.uint64) + np.array(image2, dtype=np.uint64)

def normalize_image(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_heatmap_colors(image: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)

def superimpose(image1: np.ndarray,
                image2: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0.0)

def abrir_archivo():
    ruta_archivo = filedialog.askopenfilename()
    return ruta_archivo

#-------FUNCION MAIN----------------------------------------------
def main():

    video_file = abrir_archivo()
    video_output = input("Por favor, ingrese la ruta y nombre del archivo de salida (sin extensión): ")
    video_skip = int(input("Por favor, ingrese el número de frames a omitir al inicio del video: "))
    take_every = int(input("Por favor, ingrese el valor para procesar cada n-ésimo frame: "))
    video_alpha = float(input("Por favor, ingrese el valor alpha para la superposición de imágenes del heatmap (0.0 a 1.0): "))

    capture = cv2.VideoCapture(video_file)
    background_subtractor = cv2.createBackgroundSubtractorKNN()

    read_succes, video_frame = capture.read()

    height, width, _ = video_frame.shape
    frames_number = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(video_output + ".mp4", fourcc, 30.0, (width, height))
    accumulated_image = np.zeros((height, width), np.uint8)

    count = 0

    while read_succes:
        read_succes, video_frame = capture.read()
        if read_succes:
            background_filter = background_subtractor.apply(video_frame)
            
            if count > video_skip and count % take_every == 0:
                erodated_image = apply_morph(background_filter,
                                                morph_type=cv2.MORPH_ERODE,
                                                kernel_size=(5, 5))
                accumulated_image = add_images(accumulated_image, erodated_image)
                normalized_image = normalize_image(accumulated_image)
                heatmap_image = apply_heatmap_colors(normalized_image)
                frames_merged = superimpose(heatmap_image, video_frame, video_alpha)

                cv2.imshow("Main", frames_merged)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                video.write(frames_merged)
                if count % 100 == 0:
                    print(f"Progress: {count}/{frames_number}")
            count += 1
            
    print(f"Progress: {frames_number}/{frames_number}")
    cv2.imwrite(video_output + ".png", heatmap_image)
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
