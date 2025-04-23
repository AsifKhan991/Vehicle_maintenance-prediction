import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import os
import joblib

MODEL_FILES = {
    'Random Forest': 'random_forest_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'SVC': 'svc_model.pkl'
}

le_dict_global = {}
trained_models = {}

# Load and preprocess dataset
def load_and_prepare_data():
    df = pd.read_csv("vehicle_maintenance_data.csv")
    df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])
    df['Warranty_Expiry_Date'] = pd.to_datetime(df['Warranty_Expiry_Date'])
    df['Days_Since_Last_Service'] = (pd.to_datetime("today") - df['Last_Service_Date']).dt.days
    df['Days_Until_Warranty_Expires'] = (df['Warranty_Expiry_Date'] - pd.to_datetime("today")).dt.days
    df.drop(columns=['Last_Service_Date', 'Warranty_Expiry_Date'], inplace=True)

    columns_to_exclude = [
        'Maintenance_History', 'Odometer_Reading', 'Insurance_Premium',
        'Tire_Condition', 'Brake_Condition', 'Battery_Status'
    ]
    df.drop(columns=[col for col in columns_to_exclude if col in df.columns], inplace=True)

    X = df.drop(columns=['Need_Maintenance'])
    y = df['Need_Maintenance']

    categorical_cols = X.select_dtypes(include='object').columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
    global le_dict_global
    le_dict_global = le_dict

    return X, y

def train_and_save_models(X, y):
    print("Training models...")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVC': SVC(probability=True)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILES[name])
        print(f"Saved {name} model to {MODEL_FILES[name]}")
    return models

def load_models():
    models = {}
    for name, file in MODEL_FILES.items():
        if os.path.exists(file):
            models[name] = joblib.load(file)
            print(f"Loaded {name} model from {file}")
    return models

X_data, y_data = load_and_prepare_data()
trained_models = load_models()

if len(trained_models) < 3:
    trained_models = train_and_save_models(X_data, y_data)

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Maintenance System")

    def open_predict_window(self):
        top = tk.Toplevel(self.root)
        PredictWindow(top)

    def open_add_data_window(self):
        top = tk.Toplevel(self.root)
        AddDataWindow(top)

class PredictWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Predict Maintenance")
        self.entries = {}
        self.fields = MaintenanceApp.fields
        self.build_form()

    def build_form(self):
        for idx, (field, ftype) in enumerate(self.fields.items()):
            ttk.Label(self.root, text=field).grid(row=idx, column=0, sticky=tk.W)
            if isinstance(ftype, list):
                var = tk.StringVar()
                combo = ttk.Combobox(self.root, textvariable=var, values=ftype, state='readonly')
                combo.grid(row=idx, column=1)
                self.entries[field] = var
            elif ftype == "numeric":
                var = tk.StringVar()
                entry = ttk.Entry(self.root, textvariable=var)
                entry.grid(row=idx, column=1)
                self.entries[field] = (var, entry)
            elif ftype == "date":
                date_entry = DateEntry(self.root, date_pattern='yyyy-mm-dd')
                date_entry.grid(row=idx, column=1)
                self.entries[field] = date_entry

        ttk.Label(self.root, text="Select Model").grid(row=len(self.fields), column=0, sticky=tk.W)
        self.model_choice = ttk.Combobox(self.root, values=list(trained_models.keys()), state='readonly')
        self.model_choice.grid(row=len(self.fields), column=1)

        self.result_label = ttk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.grid(row=len(self.fields)+2, columnspan=2, pady=10)

        ttk.Button(self.root, text="Check Maintenance", command=self.predict).grid(row=len(self.fields)+1, columnspan=2)

    def predict(self):
        try:
            data = {}
            # Collect data from the form fields
            for field, ftype in self.fields.items():
                if ftype == "numeric":
                    val, entry_widget = self.entries[field]
                    if not val.get().strip():
                        entry_widget.configure(background="pink")
                        self.result_label.config(text=f"Required field: {field}", foreground="red")
                        return
                    entry_widget.configure(background="white")
                    data[field] = float(val.get())
                elif ftype == "date":
                    data[field] = date.fromisoformat(self.entries[field].get())
                else:
                    val = self.entries[field].get()
                    if field == "Vehicle_Model" and val == "Sedan":
                        val = "Car"
                    data[field] = val

            # Calculate the days since the last service and days until warranty expires
            data['Days_Since_Last_Service'] = (date.today() - data['Last_Service_Date']).days
            data['Days_Until_Warranty_Expires'] = (data['Warranty_Expiry_Date'] - date.today()).days
            del data['Last_Service_Date']
            del data['Warranty_Expiry_Date']

            # Convert the input data to a DataFrame
            df_input = pd.DataFrame([data])

            # Ensure that all required columns are in the input data (matching training data columns)
            for col in X_data.columns:
                if col not in df_input.columns:
                    df_input[col] = 0  # Add missing columns with default values (e.g., 0)

            # Apply label encoding to categorical columns
            for col in df_input.columns:
                if col in le_dict_global:
                    df_input[col] = le_dict_global[col].transform(df_input[col])

            # Reorder the input data to match the training data columns
            df_input = df_input[X_data.columns]

            # Predict using the selected model
            selected_model = self.model_choice.get()
            prediction = trained_models[selected_model].predict(df_input)[0]

            # Display the result
            if prediction == 1:
                self.result_label.config(text="🚗 Maintenance Required", foreground="red")
            else:
                self.result_label.config(text="✅ No Maintenance Needed", foreground="green")

        except Exception as e:
            self.result_label.config(text=f"Error: {e}", foreground="red")
class AddDataWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Add New Vehicle Data")
        self.entries = {}
        self.fields = MaintenanceApp.fields.copy()
        self.fields['Need_Maintenance'] = ['Yes', 'No']
        self.build_form()

    def build_form(self):
        for idx, (field, ftype) in enumerate(self.fields.items()):
            ttk.Label(self.root, text=field).grid(row=idx, column=0, sticky=tk.W)
            if isinstance(ftype, list):
                var = tk.StringVar()
                combo = ttk.Combobox(self.root, textvariable=var, values=ftype, state='readonly')
                combo.grid(row=idx, column=1)
                self.entries[field] = var
            elif ftype == "numeric":
                var = tk.StringVar()
                entry = ttk.Entry(self.root, textvariable=var)
                entry.grid(row=idx, column=1)
                self.entries[field] = (var, entry)
            elif ftype == "date":
                date_entry = DateEntry(self.root, date_pattern='yyyy-mm-dd')
                date_entry.grid(row=idx, column=1)
                self.entries[field] = date_entry

        ttk.Button(self.root, text="Add and Save", command=self.save_data).grid(row=len(self.fields), columnspan=2, pady=10)

    def save_data(self):
        try:
            data = {}
            # Collect data from the form fields
            for field, ftype in self.fields.items():
                if ftype == "numeric":
                    val, entry_widget = self.entries[field]
                    if not val.get().strip():
                        entry_widget.configure(background="pink")
                        return
                    entry_widget.configure(background="white")
                    data[field] = float(val.get())
                elif ftype == "date":
                    data[field] = date.fromisoformat(self.entries[field].get())
                elif field == 'Need_Maintenance':
                    data[field] = 1 if self.entries[field].get() == 'Yes' else 0
                else:
                    val = self.entries[field].get()
                    if field == "Vehicle_Model" and val == "Sedan":
                        val = "Car"
                    data[field] = val

            # Convert the new data to a DataFrame
            df_new = pd.DataFrame([data])

            # Load the existing data
            df_existing = pd.read_csv("vehicle_maintenance_data.csv")

            # Combine the new and existing data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

            # Handle missing values (NaN)
            # Option 1: Fill NaN values with 0 (or use other strategies like mean/median imputation for numerical columns)
            df_combined.fillna(0, inplace=True)

            # Option 2: If you prefer to drop rows with NaN values, use this instead:
            # df_combined.dropna(inplace=True)

            # Save the combined data back to the CSV file
            df_combined.to_csv("vehicle_maintenance_data.csv", index=False)

            # Reload and preprocess the data
            global X_data, y_data, trained_models
            X_data, y_data = load_and_prepare_data()

            # Retrain the models with the new data
            trained_models = train_and_save_models(X_data, y_data)

            # Show success message
            messagebox.showinfo("Success", "Data added and models retrained successfully.")
            self.root.destroy()

        except Exception as e:
            messagebox.showerror("Error", str(e))


class MaintenanceApp:
    fields = {
        "Vehicle_Model": ["Truck", "Van", "Sedan", "SUV", "Motorcycle"],
        "Mileage (mile)": "numeric",
        "Number of reported issues": "numeric",
        "Vehicle_Age (yrs)": "numeric",
        "Fuel_Type": ["Petrol", "Diesel", "Electric", "Hybrid"],
        "Transmission_Type": ["Manual", "Automatic"],
        "Engine_Size (cc)": "numeric",
        "Last_Service_Date": "date",
        "Warranty_Expiry_Date": "date",
        "Owner_Type": ["First", "Second", "Third", "Fourth & Above"],
        "Service_History (Number of services)": "numeric",
        "Accident_History (number of reported accident)": "numeric",
        "Fuel_Efficiency (MPG)": "numeric"
    }

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("640x480")
    root.title("Vehicle maintenance prediction")
    app_frame = ttk.Frame(root)
    app_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    ttk.Label(app_frame, text="Choose an option from below-", font=("Arial", 12)).pack(pady=10)
    main_app = MainApp(root)
    ttk.Button(app_frame, text="Predict Maintenance", command=main_app.open_predict_window).pack(pady=10)
    ttk.Button(app_frame, text="Add New Data", command=main_app.open_add_data_window).pack(pady=10)
    root.mainloop()
