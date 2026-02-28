from scripts.data_loader import load_data
from scripts.preprocessing import preprocess
from scripts.visualize import plot_sample
from scripts.train_models import train_all_models

data_path = "data/hand_landmarks_data.csv"
raw_data = load_data(data_path)

# Optional: visualize first sample of each class
#for cls in sorted(raw_data['label'].unique()):
#    sample = raw_data[raw_data['label'] == cls].iloc[0]
#    plot_sample(sample, cls)

X, y, le = preprocess(raw_data)

svm_model, X_test, y_test = train_all_models(X, y)

print(f"Classes: {le.classes_}, Number of classes: {len(le.classes_)}")
