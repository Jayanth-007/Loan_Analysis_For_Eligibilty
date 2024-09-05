import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk

def train_model():
    global X

    # Load the dataset
    file_path = 'data.csv'
    data = pd.read_csv(file_path)

    # Handle missing values
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

    # Convert target variable 'Loan_Status' to binary (1 for 'Y', 0 for 'N')
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

    # Separate features and target variable
    X = data.drop(columns=['Loan_ID', 'Loan_Status'])
    y = data['Loan_Status']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Best Parameters:", best_params)
    print("Accuracy:", accuracy)

    return best_model

model = train_model()

def predict_loan_eligibility():
    try:
        user_data = [
            entries["entry_gender"].get().strip().capitalize(),
            entries["entry_married"].get().strip().capitalize(),
            entries["entry_dependents"].get().strip(),
            entries["entry_education"].get().strip().title(),
            entries["entry_self_employed"].get().strip().capitalize(),
            float(entries["entry_applicant_income"].get()),
            float(entries["entry_coapplicant_income"].get()),
            float(entries["entry_loan_amount"].get()),
            float(entries["entry_loan_amount_term"].get()),
            int(entries["entry_credit_history"].get()),
            entries["entry_property_area"].get().strip().capitalize()
        ]

        valid_gender = ['Male', 'Female']
        valid_married = ['Yes', 'No']
        valid_dependents = ['0', '1', '2', '3+']
        valid_education = ['Graduate', 'Not Graduate']
        valid_self_employed = ['Yes', 'No']
        valid_property_area = ['Urban', 'Semiurban', 'Rural']

        if user_data[0] not in valid_gender:
            raise ValueError("Invalid value for Gender")
        if user_data[1] not in valid_married:
            raise ValueError("Invalid value for Married")
        if user_data[2] not in valid_dependents:
            raise ValueError("Invalid value for Dependents")
        if user_data[3] not in valid_education:
            raise ValueError("Invalid value for Education")
        if user_data[4] not in valid_self_employed:
            raise ValueError("Invalid value for Self Employed")
        if user_data[10] not in valid_property_area:
            raise ValueError("Invalid value for Property Area")

        user_df = pd.DataFrame([user_data], columns=X.columns)
        prediction = model.predict(user_df)
        result = "Eligible" if prediction[0] == 1 else "Not Eligible"
        messagebox.showinfo("Prediction Result", f"The applicant is {result} for the loan.")
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

app = tk.Tk()
app.title("Bank Loan Predictor For Customers")
app.geometry("500x750")
app.configure(bg='#add8e6')  # Set background color to light blue

# Add title
title = tk.Label(app, text="Bank Loan Predictor", font=("Helvetica", 24, "bold"), bg='#add8e6', fg='#00008b')
title.pack(pady=10)

# Add bank icon
try:
    bank_icon = Image.open("bank_icon.png")
    bank_icon = bank_icon.resize((100, 100), Image.LANCZOS)
    bank_icon = ImageTk.PhotoImage(bank_icon)
    icon_label = tk.Label(app, image=bank_icon, bg='#add8e6')
    icon_label.pack(pady=10)
except Exception as e:
    print(f"Error loading image: {e}")

form_frame = tk.Frame(app, bg='#add8e6')
form_frame.pack(pady=10, padx=10)

labels_entries = [
    ("Gender", "entry_gender", ['Male', 'Female']),
    ("Married", "entry_married", ['Yes', 'No']),
    ("Dependents", "entry_dependents", ['0', '1', '2', '3+']),
    ("Education", "entry_education", ['Graduate', 'Not Graduate']),
    ("Self Employed", "entry_self_employed", ['Yes', 'No']),
    ("Applicant Income", "entry_applicant_income", []),
    ("Coapplicant Income", "entry_coapplicant_income", []),
    ("Loan Amount", "entry_loan_amount", []),
    ("Loan Amount Term", "entry_loan_amount_term", []),
    ("Credit History", "entry_credit_history", ['1', '0']),
    ("Property Area", "entry_property_area", ['Urban', 'Semiurban', 'Rural'])
]

entries = {}
for label_text, var_name, options in labels_entries:
    label = tk.Label(form_frame, text=label_text, bg='#add8e6', fg='#00008b')  # Set text color to dark blue
    label.grid(row=len(entries), column=0, sticky=tk.W, padx=10, pady=5)
    if options:
        entry = ttk.Combobox(form_frame, values=options, state="readonly")
    else:
        entry = tk.Entry(form_frame)
    entry.grid(row=len(entries), column=1, padx=10, pady=5)
    entries[var_name] = entry

predict_button = tk.Button(app, text="Predict Loan Eligibility", command=predict_loan_eligibility, bg='#00008b', fg='#ffffff')
predict_button.pack(pady=20)

app.mainloop()
