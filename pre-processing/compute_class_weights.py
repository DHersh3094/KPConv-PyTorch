from collections import Counter
import numpy as np
import sys
from sklearn.utils.class_weight import compute_class_weight

"""
Calculate class weights from the list of training files (train.txt)
python calculate_class_weights.py ../ALS_data_processed/NeuesPalaisTrees_v27/train.txt 
"""

file_path = sys.argv[1]  

with open(file_path, 'r') as file:
    basenames = [line.split('_')[0] for line in file]

class_counts = Counter(basenames)

classes = sorted(list(class_counts.keys()))

y = []

for cls, count in class_counts.items():
    y.extend([cls] * count)
    
y = np.array(y)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(classes),
    y=y
)
    
class_weights = list(class_weights)

print(f"Class counts: {class_counts}\n")
print(f'Sorted: {classes}')
print(f'Class weights: {class_weights}')
