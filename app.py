import streamlit as st
import pandas as pd
import ast


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


def field_level_view(parsed_data, field):
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

    def highlight_values(row):
        colors = [""]
        for val in row[1:]:
            if isinstance(val, str) and val.lower().startswith("n/a -"):
                colors.append("background-color: lightgrey")
            else:
                colors.append("background-color: lightgreen")
        return colors

    styled_df = df.style.apply(highlight_values, axis=1)
    return styled_df, df


def calculate_metrics(df1, df2):
    """
    Calculate confusion matrix and metrics between two models.
    df1 is considered the "truth" and df2 is the prediction.
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for val1, val2 in zip(df1, df2):
        val1_clean = val1.strip().lower()
        val2_clean = val2.strip().lower()
        
        # Check if either value is N/A
        val1_is_na = val1_clean.startswith("n/a -")
        val2_is_na = val2_clean.startswith("n/a -")
        
        # If both have values (not N/A)
        if not val1_is_na and not val2_is_na:
            if val1_clean == val2_clean:  # Both match
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

st.title("LLM Output Comparator")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    parsed_data, fields = parse_json(df)
    
    # Get model columns (exclude non-model columns)
    model_columns = [col for col in df.columns if col.lower() not in ["description", "story number"]]
    
    # Display comparison metrics for two models
    if len(model_columns) == 2:
        st.subheader(f"Comparison Metrics: {model_columns[0]} vs {model_columns[1]}")
        
        # Determine which model name is alphabetically larger to use as "truth"
        # This is a simple heuristic - adjust if needed for your specific models
        larger_model = max(model_columns)
        smaller_model = min(model_columns)
        
        st.write(f"Using **{larger_model}** as the reference model")
        
        all_field_metrics = {}
        
        # Calculate metrics for each field
        for field in fields:
            _, field_df = field_level_view(parsed_data, field)
            
            metrics = calculate_metrics(
                field_df[larger_model], 
                field_df[smaller_model]
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
        _, field_df = field_level_view(parsed_data, field)
        summary_data[field] = field_df.iloc[:, 1:].apply(lambda col: (col != "N/A - Missing").sum()).to_dict()

    summary_df = pd.DataFrame(summary_data).T
    st.dataframe(summary_df)

    st.subheader("Field Level View for All Fields")
    for field in fields:
        st.write(f"### Field: {field}")
        styled_df, field_df = field_level_view(parsed_data, field)
        st.write(styled_df)

        non_na_counts = field_df.iloc[:, 1:].apply(lambda col: col.map(lambda x: not x.lower().startswith("n/a -")).sum())
        st.write("#### Count of Non-NA Values:")
        st.write(non_na_counts.to_frame().T)
