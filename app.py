import streamlit as st
import pandas as pd
import ast
from difflib import SequenceMatcher
import os


# Initialize session state variables
if 'matching_mode' not in st.session_state:
    st.session_state.matching_mode = "Exact Match"
if 'files_data' not in st.session_state:
    st.session_state.files_data = {}  # Store multiple file data
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None


def is_no_value_message(text, field_name=None):
    """
    Check if text represents a "no value" or error message.
    Returns True if it matches common no-value patterns.
    """
    text = str(text).lower().strip()
    
    # List of common phrases indicating no value
    no_value_phrases = [
        "not available", "none", "not specified", "null", "not disclosed", "n/a", 
        "no value", "missing", "unknown", "no data", "not found", "unavailable"
    ]
    
    # Check for exact matches
    if any(phrase == text for phrase in no_value_phrases):
        return True
    
    # Check for pattern matches
    if field_name:
        field_lower = field_name.lower()
        no_value_patterns = [
            f"no {field_lower}", 
            f"{field_lower} not",
            f"missing {field_lower}", 
            f"could not find {field_lower}"
        ]
        if any(pattern in text for pattern in no_value_patterns):
            return True
    
    # Check for starts with patterns
    no_value_starts = ["no ", "not ", "missing ", "n/a", "unknown "]
    if any(text.startswith(start) for start in no_value_starts):
        return True
        
    return False


def are_values_similar(val1, val2, field_name=None):
    """
    Compare two values based on the selected matching mode.
    """
    matching_mode = st.session_state.matching_mode
    val1 = str(val1).strip().lower()
    val2 = str(val2).strip().lower()
    
    # Always return true for exact matches
    if val1 == val2:
        return True
    
    # If exact matching only
    if matching_mode == "Exact Match":
        return False
    
    # If using Fuzzy Match
    if matching_mode == "Fuzzy Match":
        # Check if both are "no value" messages
        if is_no_value_message(val1, field_name) and is_no_value_message(val2, field_name):
            return True
            
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, val1, val2).ratio()
        if similarity > 0.8:  # Threshold can be adjusted
            return True
    
    return False


def load_data(file):
    df = pd.read_csv(file, skipinitialspace=True, engine='python')
    return df


def normalize_response(val):
    """
    Parses a value that might be a stringified dict, list of dicts, etc.
    Returns a merged dict without 'description' keys.
    """
    if pd.isna(val):
        return {}

    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, dict):
            parsed.pop("description", None)
            return parsed
        elif isinstance(parsed, list):
            combined = {}
            for item in parsed:
                if isinstance(item, dict):
                    item.pop("description", None)
                    combined.update(item)
            return combined
    except Exception:
        return {}

    return {}


def parse_json(df):
    parsed_data = {}
    fields = set()

    for model in df.columns:
        if model.lower() in ["description", "story number"]:
            continue

        parsed_data[model] = []
        for val in df[model]:
            clean = normalize_response(val)
            parsed_data[model].append(clean)
            fields.update(clean.keys())

    fields.discard("description")
    return parsed_data, sorted(fields)


def field_level_view(parsed_data, field, show_only_differences=False):
    """
    Generate a view of field values across models.
    If show_only_differences is True, only show rows where models differ.
    """
    def format_value(val):
        """
        Normalize and convert all values to a list of cleaned strings.
        Handles known "unavailable" values and formats them.
        """
        unavailable_values = {'not available', 'none', 'not specified', 'null', 'not disclosed', ''}
        if isinstance(val, str):
            if val.strip().lower() in unavailable_values:
                return [f"N/A - {val.strip() or 'Blank'}"]
            return [val.strip()]
        elif isinstance(val, list):
            result = []
            for v in val:
                if isinstance(v, str) and v.strip().lower() in unavailable_values:
                    result.append(f"N/A - {v.strip() or 'Blank'}")
                else:
                    result.append(str(v))
            return result
        elif pd.isna(val):
            return ["N/A - Missing"]
        else:
            return [str(val)]

    result = {"Story Number": list(range(len(next(iter(parsed_data.values())))))}
    for model, responses in parsed_data.items():
        result[model] = [", ".join(format_value(response.get(field, "N/A"))) for response in responses]

    df = pd.DataFrame(result)
    
    # Filter rows to show only differences if requested
    if show_only_differences and len(df.columns) > 2:  # At least one model column
        models = [col for col in df.columns if col != "Story Number"]
        if len(models) == 2:
            # Create a mask for rows where values differ
            mask = []
            for i in range(len(df)):
                val1 = df[models[0]].iloc[i]
                val2 = df[models[1]].iloc[i]
                are_similar = are_values_similar(val1, val2, field)
                mask.append(not are_similar)  # Keep rows where values are NOT similar
                
            # Apply the mask to show only different rows
            if any(mask):  # Only filter if there are differences
                df = df[mask].reset_index(drop=True)
            elif all(not m for m in mask):  # No differences
                df = pd.DataFrame({"Story Number": [], models[0]: [], models[1]: []})

    # Define highlighting function based on number of model columns
    def highlight_values(row):
        if len(row) <= 2:  # Story Number and maybe 1 model
            return [""] * len(row)
            
        # If we have multiple models, highlight differences
        if len(row) == 3:  # Story Number + 2 models
            val1 = row.iloc[1]
            val2 = row.iloc[2]
            
            # Check if values are similar based on current matching mode
            are_similar = are_values_similar(val1, val2, field)
            
            if are_similar:
                return ["", "background-color: lightgreen", "background-color: lightgreen"]
            else:
                return ["", "background-color: lightsalmon", "background-color: lightsalmon"]
        else:
            # For 3+ models, use original N/A highlighting
            colors = [""]
            for val in row[1:]:
                if isinstance(val, str) and val.lower().startswith("n/a -"):
                    colors.append("background-color: lightgrey")
                else:
                    colors.append("background-color: lightgreen")
            return colors

    styled_df = df.style.apply(highlight_values, axis=1)
    return styled_df, df


def calculate_metrics(df1, df2, field_name=None):
    """
    Calculate confusion matrix and metrics between two models.
    df1 is considered the "truth" and df2 is the prediction.
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for val1, val2 in zip(df1, df2):
        val1_is_na = val1.strip().lower().startswith("n/a -")
        val2_is_na = val2.strip().lower().startswith("n/a -")
        
        # Compare values based on matching mode
        values_match = are_values_similar(val1, val2, field_name)
        
        # If both have values (not N/A)
        if not val1_is_na and not val2_is_na:
            if values_match:  # Both match
                true_positive += 1
            else:  # Different values
                false_negative += 1
        # If truth has value but prediction doesn't
        elif not val1_is_na and val2_is_na:
            false_negative += 1
        # If truth has no value but prediction does
        elif val1_is_na and not val2_is_na:
            false_positive += 1
        # If both have no values
        else:
            true_negative += 1
    
    # Calculate metrics
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total > 0 else 0
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "True Positive": true_positive,
        "True Negative": true_negative,
        "False Positive": false_positive,
        "False Negative": false_negative,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    return metrics


# Function to handle matching mode change
def update_matching_mode():
    st.session_state.matching_mode = st.session_state.temp_matching_mode


# Function to handle file upload and processing
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        # Load and process data
        df = load_data(uploaded_file)
        parsed_data, fields = parse_json(df)
        
        # Store in session state
        st.session_state.files_data[file_name] = {
            "df": df,
            "parsed_data": parsed_data,
            "fields": fields
        }
        
        # Set as selected file if this is the first file
        if st.session_state.selected_file is None:
            st.session_state.selected_file = file_name


# Streamlit UI
st.set_page_config(page_title="LLM Output Comparator", layout="wide")
st.sidebar.title("LLM Output Comparator")

# Add matching mode selection in sidebar
st.sidebar.subheader("Matching Mode Settings")
st.session_state.temp_matching_mode = st.sidebar.radio(
    "Select Matching Mode:",
    ["Exact Match", "Fuzzy Match"],
    key="matching_mode_radio",
    on_change=update_matching_mode,
    index=["Exact Match", "Fuzzy Match"].index(st.session_state.matching_mode 
           if st.session_state.matching_mode in ["Exact Match", "Fuzzy Match"] else "Exact Match"),
    help="""
    Exact Match: Values must be exactly the same.
    Fuzzy Match: Similar error messages are considered the same using rules and string similarity.
    """
)

# Add "Show only differences" checkbox to sidebar
show_only_differences = st.sidebar.checkbox("Show only differences", value=False)

st.title("LLM Output Comparator")

# File uploader section
st.sidebar.subheader("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")

# Process newly uploaded file
process_uploaded_file(uploaded_file)

# Display list of uploaded files
if st.session_state.files_data:
    st.sidebar.subheader("Uploaded Files")
    file_names = list(st.session_state.files_data.keys())
    selected_file = st.sidebar.selectbox(
        "Select file to analyze:",
        file_names,
        key="file_selector",
        index=file_names.index(st.session_state.selected_file) if st.session_state.selected_file in file_names else 0
    )
    
    # Update selected file
    st.session_state.selected_file = selected_file
    
    # Clear a specific file if requested
    if st.sidebar.button("Remove Selected File"):
        if selected_file in st.session_state.files_data:
            del st.session_state.files_data[selected_file]
            if not st.session_state.files_data:  # If no files left
                st.session_state.selected_file = None
            else:  # Select first available file
                st.session_state.selected_file = next(iter(st.session_state.files_data))
            st.experimental_rerun()

    # Clear all files if requested
    if st.sidebar.button("Clear All Files"):
        st.session_state.files_data = {}
        st.session_state.selected_file = None
        st.experimental_rerun()

# Display content for selected file
if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files_data:
    file_data = st.session_state.files_data[st.session_state.selected_file]
    df = file_data["df"]
    parsed_data = file_data["parsed_data"]
    fields = file_data["fields"]
    
    # Get model columns (exclude non-model columns)
    model_columns = [col for col in df.columns if col.lower() not in ["description", "story number"]]
    
    # Display current matching mode
    st.write(f"Current matching mode: **{st.session_state.matching_mode}**")
    
    # Display selected file name
    st.write(f"Analyzing file: **{st.session_state.selected_file}**")
    
    # Display comparison metrics for two models
    if len(model_columns) == 2:
        st.subheader(f"Comparison Metrics: {model_columns[0]} vs {model_columns[1]}")
        
        # Determine which model name is alphabetically larger to use as "truth"
        larger_model = max(model_columns)
        smaller_model = min(model_columns)
        
        st.write(f"Using **{larger_model}** as the reference model")
        
        all_field_metrics = {}
        
        # Calculate metrics for each field
        for field in fields:
            _, field_df = field_level_view(parsed_data, field, False)
            
            metrics = calculate_metrics(
                field_df[larger_model], 
                field_df[smaller_model],
                field
            )
            
            all_field_metrics[field] = metrics
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame(all_field_metrics).T
        
        # Format percentage columns
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.2%}")
            
        st.dataframe(metrics_df)

    st.subheader("Summary of Facts Count")
    summary_data = {}
    for field in fields:
        _, field_df = field_level_view(parsed_data, field, False)
        summary_data[field] = field_df.iloc[:, 1:].apply(lambda col: (col != "N/A - Missing").sum()).to_dict()

    summary_df = pd.DataFrame(summary_data).T
    st.dataframe(summary_df)

    # Display field-level view using tabs
    st.subheader("Field Level View")
    
    # Create tabs for fields
    tabs = st.tabs([f"{field}" for field in fields])
    
    # Create content for each tab
    for i, field in enumerate(fields):
        with tabs[i]:
            styled_df, field_df = field_level_view(parsed_data, field, show_only_differences)
            st.write(styled_df)

            non_na_counts = field_df.iloc[:, 1:].apply(lambda col: col.map(lambda x: not x.lower().startswith("n/a -")).sum())
            st.write("#### Count of Non-NA Values:")
            st.write(non_na_counts.to_frame().T)
