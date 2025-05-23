import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("vehicle_maintenance_data.csv")

# Preprocess dates
df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])
df['Warranty_Expiry_Date'] = pd.to_datetime(df['Warranty_Expiry_Date'])
df['Days_Since_Last_Service'] = (pd.to_datetime("today") - df['Last_Service_Date']).dt.days
df['Days_Until_Warranty_Expires'] = (df['Warranty_Expiry_Date'] - pd.to_datetime("today")).dt.days
df.drop(columns=['Last_Service_Date', 'Warranty_Expiry_Date'], inplace=True)

# Drop specified columns
columns_to_exclude = [
    'Maintenance_History', 'Odometer_Reading', 'Insurance_Premium',
    'Tire_Condition', 'Brake_Condition', 'Battery_Status'
]
df.drop(columns=columns_to_exclude, inplace=True)

# Split features and target
X = df.drop(columns=['Need_Maintenance'])
y = df['Need_Maintenance']

# Encode categorical variables
categorical_cols = X.select_dtypes(include='object').columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='mediumseagreen')
plt.xlabel("Importance Score")
plt.title("Feature Importance (Excluding Specified Columns)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
