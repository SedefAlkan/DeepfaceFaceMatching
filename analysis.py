import pandas as pd
import os
from deepface import DeepFace
from tqdm import tqdm
import torch
import tensorflow as tf


models = [
    "VGG-Face",
    "Facenet",
    "ArcFace",
    "Dlib",
    "GhostFaceNet"
]


detectors = [
    "retinaface",
    "mtcnn",
    "dlib",
    "ssd"
]


dataset_path = "dataset"


df = pd.read_csv("master.csv")


results = []


if tf.config.list_physical_devices('GPU'):
    print("GPU kullanılabilir. GPU üzerinde çalıştırılacak.")
    device_name = "/GPU:0"
else:
    print("GPU kullanılamaz, CPU üzerinde çalıştırılacak.")
    device_name = "/CPU:0"


with tf.device(device_name):

    for model in models:
        for detector in detectors:
            correct_predictions = 0  # Doğru tahmin sayısı
            total_samples = 0        # Toplam örnek sayısı

            print(f"\nModel: {model}, Detector: {detector}\n")

         
            for index, row in tqdm(df.iterrows(), total=len(df)):
                img1_path = os.path.join(dataset_path, row["file_x"])
                img2_path = os.path.join(dataset_path, row["file_y"])
                actual_label = row["Decision"]  # CSV'deki gerçek etiket ("Yes" / "No")

                try:
                    
                    result = DeepFace.verify(img1_path, img2_path, 
                                             model_name=model, 
                                             detector_backend=detector)

                    predicted_label = "Yes" if result["verified"] else "No"

                    
                    if predicted_label == actual_label:
                        correct_predictions += 1

                    total_samples += 1

                except Exception as e:
                    print(f"Hata: {img1_path} - {img2_path} -> {e}")

            # Doğruluk oranı hesapla
            accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

            
            results.append([model, detector, accuracy])

           
            print(f"Doğruluk Oranı: {accuracy:.2f}%")

    
    result_df = pd.DataFrame(results, columns=["Model", "Detector", "Accuracy"])

    
    result_df.to_csv("deepface_comparison_results.csv", index=False)

print("\nTüm karşılaştırmalar tamamlandı. Sonuçlar 'deepface_comparison_results.csv' dosyasına kaydedildi.")
