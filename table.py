import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("deepface_comparison_results.csv")


df_pivot = df.pivot(index="Detector", columns="Model", values="Accuracy")


ordered_detectors = ["retinaface", "mtcnn", "dlib", "ssd"]
ordered_models = ["ArcFace", "VGG-Face", "Dlib", "Facenet", "GhostFaceNet"]

# Mevcut sütunları ve indeksleri sıralamaya uygun hale getirme
df_pivot = df_pivot.loc[ordered_detectors, ordered_models]


fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_pivot.values,
                 colLabels=df_pivot.columns,
                 rowLabels=df_pivot.index,
                 cellLoc="center",
                 loc="center")


plt.title("Model Detector Accuracy", fontsize=14, fontweight="bold")


plt.show()
