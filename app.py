import streamlit as st
import pandas as pd
import ast
import re
from difflib import SequenceMatcher
import requests
import json
import time
import os


# Initialize session state for important variables
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = "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"  # Default API key

if 'matching_mode' not in st.session_state:
    st.session_state.matching_mode = "Exact Match"

if "hf_cache" not in st.session_state:
    st.session_state.hf_cache = {}


# Hugging Face API functions
def compare_with_huggingface(text1, text2, field_name, api_key):
    """
    Use Hugging Face's Inference API with GPT2 to compare text values.
    Returns True if the model determines they are semantically equivalent.
    """
    # Create a cache key to avoid redundant API calls
    cache_key = f"{text1.lower()}|||{text2.lower()}|||{field_name.lower()}"
    
    # Return cached result if available
    if cache_key in st.session_state.hf_cache:
        return st.session_state.hf_cache[cache_key]
    
    # Set up the API request
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Construct the prompt for semantic comparison
    prompt = f"""
    Question: Are these two values for the field "{field_name}" semantically equivalent?
    Value 1: "{text1}"
    Value 2: "{text2}"
    
    If both values indicate that the information is not available or missing (like "not found", "no value", etc.), 
    they should be considered equivalent.
    
    Answer (yes/no):
    """
    
    try:
        # Make the API request
        payload = {"inputs": prompt, "parameters": {"max_length": 50}}
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = result.get("generated_text", "")
        
        # Extract yes/no from the generated text
        lower_text = generated_text.lower()
        
        # Look for yes/no indicators in the response
        is_equivalent = ("yes" in lower_text and "no " not in lower_text[:lower_text.find("yes")+5]) or \
                        ("equivalent" in lower_text) or \
                        ("same" in lower_text and "not same" not in lower_text)
        
        # Cache the result
        st.session_state.hf_cache[cache_key] = is_equivalent
        return is_equivalent
        
    except Exception as e:
        st.warning(f"Hugging Face API error: {str(e)}. Falling back to basic comparison.")
        # Fall back to basic comparison
        return are_values_similar_basic(text1, text2, field_name, True)


def is_no_value_message(text, field_name=None):
    """
    Check if text represents a "no value" or error message.
    Returns True if it matches common no-value patterns.
    """
    text = text.lower().strip()
    
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


def are_values_similar_basic(val1, val2, field_name=None, fuzzy_match=False):
    """
    Compare two values to determine if they are semantically equivalent using basic methods.
    """
    val1 = str(val1).strip().lower()
    val2 = str(val2).strip().lower()
    
    # If exactly the same, return True
    if val1 == val2:
        return True
        
    # If fuzzy matching is enabled
    if fuzzy_match:
        # Check if both are "no value" messages
        if is_no_value_message(val1, field_name) and is_no_value_message(val2, field_name):
            return True
            
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, val1, val2).ratio()
        if similarity > 0.8:  # Threshold can be adjusted
            return True
            
    return False


def are_values_similar(val1, val2, field_name=None, matching_mode=None, api_key=None):
    """
    Compare two values based on the selected matching mode.
    """
    # Use session state matching mode if not provided
    if matching_mode is None:
        matching_mode = st.session_state.matching_mode
        
    val1 = str(val1).strip()
    val2 = str(val2).strip()
    
    # Always return true for exact matches
    if val1.lower() == val2.lower():
        return True
    
    # If exact matching only
    if matching_mode == "Exact Match":
        return False
    
    # If using Basic Fuzzy Match
    if matching_mode == "Basic Fuzzy Match":
        return are_values_similar_basic(val1, val2, field_name, True)
    
    # If using Hugging Face GPT2
    if matching_mode == "Hugging Face GPT2" and api_key:
        return compare_with_huggingface(val1, val2, field_name, api_key)
    
    # Fall back to basic fuzzy matching
    return are_values_similar_basic(val1, val2, field_name, True)


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


def field_level_view(parsed_data, field, matching_mode=None, show_only_differences=False, api_key=None):
    """
    Generate a view of field values across models.
    If show_only_differences is True, only show rows where models differ.
    """
    # Use session state matching mode if not provided
    if matching_mode is None:
        matching_mode = st.session_state.matching_mode
        
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
                are_similar = are_values_similar(val1, val2, field, matching_mode, api_key)
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
            are_similar = are_values_similar(val1, val2, field, matching_mode, api_key)
            
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


def calculate_metrics(df1, df2, field_name=None, matching_mode=None, api_key=None):
    """
    Calculate confusion matrix and metrics between two models.
    df1 is considered the "truth" and df2 is the prediction.
    """
    # Use session state matching mode if not provided
    if matching_mode is None:
        matching_mode = st.session_state.matching_mode
        
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for val1, val2 in zip(df1, df2):
        val1_is_na = val1.strip().lower().startswith("n/a -")
        val2_is_na = val2.strip().lower().startswith("n/a -")
        
        # Compare values based on matching mode
        values_match = are_values_similar(val1, val2, field_name, matching_mode, api_key)
        
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


# Streamlit UI
st.set_page_config(page_title="Fact Comparison by Field", layout="wide")
st.sidebar.title("Fact Comparison by Field")

# Function to handle matching mode change
def update_matching_mode():
    st.session_state.matching_mode = st.session_state.temp_matching_mode
    # Clear the cache when changing modes
    st.session_state.hf_cache = {}

# Add matching mode selection in sidebar
st.sidebar.subheader("Matching Mode Settings")
st.session_state.temp_matching_mode = st.sidebar.radio(
    "Select Matching Mode:",
    ["Exact Match", "Basic Fuzzy Match", "Hugging Face GPT2"],
    key="matching_mode_radio",
    on_change=update_matching_mode,
    index=["Exact Match", "Basic Fuzzy Match", "Hugging Face GPT2"].index(st.session_state.matching_mode),
    help="""
    Exact Match: Values must be exactly the same.
    Basic Fuzzy Match: Similar error messages are considered the same using rules and string similarity.
    Hugging Face GPT2: Uses Hugging Face's GPT2 model to determine semantic similarity.
    """
)

# Show API key configuration when Hugging Face GPT2 is selected
if st.session_state.matching_mode == "Hugging Face GPT2":
    with st.sidebar.expander("Hugging Face API Settings", expanded=True):
        new_api_key = st.text_input(
            "API Key",
            value=st.session_state.hf_api_key,
            type="password",
            key="hf_api_key_input",
            help="Enter your Hugging Face API key"
        )
        
        # Update API key if changed
        if new_api_key != st.session_state.hf_api_key:
            st.session_state.hf_api_key = new_api_key
            # Clear cache when API key changes
            st.session_state.hf_cache = {}
            
        st.info("Using GPT2 model to compare text semantically")

st.title("LLM Output Comparator")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    parsed_data, fields = parse_json(df)
    
    # Get model columns (exclude non-model columns)
    model_columns = [col for col in df.columns if col.lower() not in ["description", "story number"]]
    
    # Add filter option for two-model comparisons
    show_only_differences = False
    if len(model_columns) == 2:
        show_only_differences = st.checkbox("Show only differences", value=False)
    
    # Get API key if using Hugging Face
    api_key = st.session_state.hf_api_key if st.session_state.matching_mode == "Hugging Face GPT2" else None
    
    # Display current matching mode
    st.write(f"Current matching mode: **{st.session_state.matching_mode}**")
    
    # Display comparison metrics for two models
    if len(model_columns) == 2:
        st.subheader(f"Comparison Metrics: {model_columns[0]} vs {model_columns[1]}")
        
        # Determine which model name is alphabetically larger to use as "truth"
        larger_model = max(model_columns)
        smaller_model = min(model_columns)
        
        st.write(f"Using **{larger_model}** as the reference model")
        
        # Status message for API usage
        if st.session_state.matching_mode == "Hugging Face GPT2":
            status_msg = st.info("⏳ Processing with Hugging Face API - this may take a moment...")
        
        all_field_metrics = {}
        
        # Calculate metrics for each field
        for field in fields:
            _, field_df = field_level_view(parsed_data, field, None, False, api_key)
            
            metrics = calculate_metrics(
                field_df[larger_model], 
                field_df[smaller_model],
                field,
                None,
                api_key
            )
            
            all_field_metrics[field] = metrics
        
        # Clear status message if present
        if st.session_state.matching_mode == "Hugging Face GPT2" and 'status_msg' in locals():
            status_msg.empty()
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame(all_field_metrics).T
        
        # Format percentage columns
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.2%}")
            
        st.dataframe(metrics_df)

    st.subheader("Summary of Facts Count")
    summary_data = {}
    for field in fields:
        _, field_df = field_level_view(parsed_data, field, None, False, api_key)
        summary_data[field] = field_df.iloc[:, 1:].apply(lambda col: (col != "N/A - Missing").sum()).to_dict()

    summary_df = pd.DataFrame(summary_data).T
    st.dataframe(summary_df)

    st.subheader("Field Level View for All Fields")
    for field in fields:
        st.write(f"### Field: {field}")
        
        # Show Hugging Face processing status if applicable
        if st.session_state.matching_mode == "Hugging Face GPT2" and show_only_differences:
            status = st.empty()
            status.info("⏳ Processing with Hugging Face API...")
        
        styled_df, field_df = field_level_view(parsed_data, field, None, show_only_differences, api_key)
        
        # Clear status message
        if st.session_state.matching_mode == "Hugging Face GPT2" and show_only_differences and 'status' in locals():
            status.empty()
            
        st.write(styled_df)

        non_na_counts = field_df.iloc[:, 1:].apply(lambda col: col.map(lambda x: not x.lower().startswith("n/a -")).sum())
        st.write("#### Count of Non-NA Values:")
        st.write(non_na_counts.to_frame().T)
