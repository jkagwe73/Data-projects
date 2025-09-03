import gradio as gr
import pandas as pd
import joblib
import os # To access environment variables for HF_TOKEN if needed

# --- Configuration ---
MODEL_PATH = "loan_default_model.pkl" # Make sure this matches your file name
#PREPROCESSOR_PATH = "preprocessor.pkl" # If you have one
FEATURE_COLUMNS_PATH = "feature_columns.pkl" # Make sure this matches

# --- Global Variables for Model & Features (loaded once at startup) ---
model = None
FEATURE_COLUMNS = None
DEFAULT_THRESHOLD = 0.3 # Your default threshold

# --- Load your pre-trained model and preprocessing artifacts ---
# This block runs once when the app starts up
try:
    model = joblib.load(MODEL_PATH)
    FEATURE_COLUMNS = joblib.load(FEATURE_COLUMNS_PATH)
    print("Model and feature columns loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Required file not found: {e}. Make sure '{MODEL_PATH}' and '{FEATURE_COLUMNS_PATH}' are in your Space.")
    # Exit or raise an error to prevent the app from crashing in a user-unfriendly way
    # In a deployed app, you might want to log this and show a friendly message.
except Exception as e:
    print(f"Error loading assets: {e}")

# --- Dummy DataFrame for Gradio Choices (IMPORTANT: this simulates your original data) ---
# This part is crucial if you're using df['Column'].unique().tolist() for choices
# You need to ensure 'df' exists and is loaded correctly.
# A better practice for deployed apps is to save and load these unique choices as a separate file
# (e.g., a JSON or PKL) if the original `df` is large or not needed for anything else.
# For demonstration, we'll assume `df` is loaded from a CSV.
try:
    df = pd.read_csv("Loan_default_5000.csv") # Load your CSV here
    # You might need to adjust this path depending on where the CSV is in your Space
    print("Original DataFrame loaded for choices.")
except FileNotFoundError:
    print("Warning: Loan_default_5000.csv not found. Gradio dropdown choices might be empty.")
    # Create a dummy df or define choices manually if the file isn't essential for prediction logic
    df = pd.DataFrame({
        'Education': ['High School', 'University', 'Graduate'],
        'EmploymentType': ['Salaried', 'Self-Employed', 'Business'],
        'MaritalStatus': ['Single', 'Married', 'Divorced'],
        'LoanPurpose': ['Home', 'Car', 'Education', 'Debt Consolidation', 'Other'],
        'HasMortgage': ['Yes', 'No'],
        'HasDependents': ['Yes', 'No'],
        'HasCoSigner': ['Yes', 'No']
    })
except Exception as e:
    print(f"Error loading dummy DataFrame: {e}")
    # Handle the error by creating a minimal DataFrame to avoid crashing
    df = pd.DataFrame({
        'Education': ['High School', 'University', 'Graduate'],
        'EmploymentType': ['Salaried', 'Self-Employed', 'Business'],
        'MaritalStatus': ['Single', 'Married', 'Divorced'],
        'LoanPurpose': ['Home', 'Car', 'Education', 'Debt Consolidation', 'Other'],
        'HasMortgage': ['Yes', 'No'],
        'HasDependents': ['Yes', 'No'],
        'HasCoSigner': ['Yes', 'No']
    })


# Helper function to ensure a value is a scalar, handling potential single-element list wrapping
def _ensure_scalar(value):
    """
    Ensures a value is a scalar. If it's a list with a single element, it unwraps it.
    This helps prevent TypeError: unhashable type: 'list' when get_dummies
    receives a list wrapped around a string from UI inputs.
    """
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value

# Define the prediction function
def predict_loan_default(
    Age, Income, LoanAmount, CreditScore, MonthsEmployed,
    NumCreditLines, InterestRate, LoanTerm, DTIRatio,
    Education, EmploymentType, MaritalStatus,
    HasMortgage, HasDependents, LoanPurpose, HasCoSigner
):
    # Create a dataframe from the input data, ensuring scalar values
    input_data_dict = {
        'Age': _ensure_scalar(Age),
        'Income': _ensure_scalar(Income),
        'LoanAmount': _ensure_scalar(LoanAmount),
        'CreditScore': _ensure_scalar(CreditScore),
        'MonthsEmployed': _ensure_scalar(MonthsEmployed),
        'NumCreditLines': _ensure_scalar(NumCreditLines),
        'InterestRate': _ensure_scalar(InterestRate),
        'LoanTerm': _ensure_scalar(LoanTerm),
        'DTIRatio': _ensure_scalar(DTIRatio),
        'Education': _ensure_scalar(Education),
        'EmploymentType': _ensure_scalar(EmploymentType),
        'MaritalStatus': _ensure_scalar(MaritalStatus),
        'HasMortgage': _ensure_scalar(HasMortgage),
        'HasDependents': _ensure_scalar(HasDependents),
        'LoanPurpose': _ensure_scalar(LoanPurpose),
        'HasCoSigner': _ensure_scalar(HasCoSigner)
    }
    input_data = pd.DataFrame([input_data_dict]) # Wrap the dict in a list to create a single row DataFrame

    # one-hot encode and align the input data with the model's expected feature set
    # Ensure categorical columns exist in the DataFrame before get_dummies if they might be missing
    categorical_cols_to_dummies = [
        'Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose'
    ]

    # Filter for columns that are actually in input_data_dict
    cols_present_and_categorical = [col for col in categorical_cols_to_dummies if col in input_data.columns]


    input_data = pd.get_dummies(input_data, columns=cols_present_and_categorical, drop_first=True)

    # Ensure the input data has the same columns as the model
    # Use reindex with `columns` parameter and `fill_value` for robustness
    if FEATURE_COLUMNS is None:
        return "Error: Model features not loaded. Cannot make prediction."
    input_data = input_data.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Make the prediction
    if model is None:
        return "Error: Model not loaded. Cannot make prediction."

    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Apply the threshold to get the final prediction
    prediction = (prediction_proba > DEFAULT_THRESHOLD).astype(int)

    return "Default" if prediction[0] == 1 else "No Default"

# Build Gradio interface
def build_interface():
    # Define inputs for Gradio. The arguments `minimum` and `maximum` are required for Slider.
    # The `choices` argument is required for Dropdown.
    # Ensure all categorical columns are handled correctly.
    # Note: df must be loaded for `.unique().tolist()` to work.

    # Check if df is properly loaded before using its unique values
    education_choices = df['Education'].unique().tolist() if 'Education' in df.columns else ['High School', 'University', 'Graduate']
    employment_choices = df['EmploymentType'].unique().tolist() if 'EmploymentType' in df.columns else ['Salaried', 'Self-Employed', 'Unemployed']
    marital_choices = df['MaritalStatus'].unique().tolist() if 'MaritalStatus' in df.columns else ['Single', 'Married', 'Divorced']
    loan_purpose_choices = df['LoanPurpose'].unique().tolist() if 'LoanPurpose' in df.columns else ['Home', 'Car', 'Education', 'Debt Consolidation', 'Other']

    with gr.Blocks() as demo:
        gr.Markdown("# Loan Default Prediction")
        gr.Markdown("## Enter the loan details to predict if the loan will default or not.")
        with gr.Row():
            Age = gr.Slider(minimum=18, maximum=100, step=1, label="Age", value=30) # Added default value
            Income = gr.Slider(minimum=0, maximum=1000000, step=1000, label="Income", value=50000)
            LoanAmount = gr.Slider(minimum=0, maximum=1000000, step=1000, label="Loan Amount", value=10000)
            CreditScore = gr.Slider(minimum=300, maximum=850, step=1, label="Credit Score", value=700)
        with gr.Row():
            MonthsEmployed = gr.Slider(minimum=0, maximum=120, step=1, label="Months Employed", value=60)
            NumCreditLines = gr.Slider(minimum=0, maximum=50, step=1, label="Number of Credit Lines", value=3)
            InterestRate = gr.Slider(minimum=0, maximum=50, step=0.1, label="Interest Rate", value=7.5)
            LoanTerm = gr.Slider(minimum=1, maximum=30, step=1, label="Loan Term (Years)", value=5)
        with gr.Row():
            DTIRatio = gr.Slider(minimum=0, maximum=100, step=1, label="Debt-to-Income Ratio (%)", value=30)
            Education = gr.Dropdown(
                choices=education_choices,
                label="Education Level",
                value=education_choices[0] if education_choices else None # Added default value
            )
            EmploymentType = gr.Dropdown(
                choices=employment_choices,
                label="Employment Type",
                value=employment_choices[0] if employment_choices else None # Added default value
            )
            MaritalStatus = gr.Dropdown(
                choices=marital_choices,
                label="Marital Status",
                value=marital_choices[0] if marital_choices else None # Added default value
            )
        with gr.Row():
            HasMortgage = gr.Dropdown(
                choices=['Yes', 'No'],
                label="Has Mortgage",
                value='No' # Added default value
            )
            HasDependents = gr.Dropdown(
                choices=['Yes', 'No'],
                label="Has Dependents",
                value='No' # Added default value
            )
            LoanPurpose = gr.Dropdown(
                choices=loan_purpose_choices,
                label="Loan Purpose",
                value=loan_purpose_choices[0] if loan_purpose_choices else None # Added default value
            )
            HasCoSigner = gr.Dropdown(
                choices=['Yes', 'No'],
                label="Has Co-Signer",
                value='No' # Added default value
            )
        with gr.Row():
            submit_button = gr.Button("Submit")
            result = gr.Textbox(label="Prediction")

        submit_button.click(
            predict_loan_default,
            inputs=[
                Age, Income, LoanAmount, CreditScore, MonthsEmployed,
                NumCreditLines, InterestRate, LoanTerm, DTIRatio,
                Education, EmploymentType, MaritalStatus,
                HasMortgage, HasDependents, LoanPurpose, HasCoSigner
            ],
            outputs=result
        )
    return demo

# --- Main execution block for the Gradio app ---
# This ensures the app only runs when executed directly
if __name__ == "__main__":
    # Check if model and FEATURE_COLUMNS were loaded successfully before building and launching
    #if model is not None and FEATURE_COLUMNS is not None:
        demo = build_interface()
        demo.launch() # Removed share=True as it's not needed on HF Spaces

        print("Application cannot start: Model or feature columns failed to load.")
