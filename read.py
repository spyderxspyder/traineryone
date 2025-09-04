import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Excel file
df = pd.read_excel(r"your_data.xlsx")   # or pd.read_csv("file.csv")

# Separate features and labels
X = df.iloc[:, :-1].values   # all columns except last as features
y = df.iloc[:, -1].values    # last column as target

# Encode labels if categorical (e.g. strings -> integers)
if y.dtype == object or str(y.dtype).startswith("str"):
    y = LabelEncoder().fit_transform(y)

# Normalize features
X = StandardScaler().fit_transform(X)

# One-hot encode labels for Keras
y = tf.keras.utils.to_categorical(y)
