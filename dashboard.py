import model_builder
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils import fomo_level
from constants import (
  FOMO_TEST_FILE,
  FEATURE_COLS,
  MODEL_DIR,
  FOMOScoreThresholds as th_fomo_score
)


# Page config
st.set_page_config(page_title="FOMO Investor Dashboard", layout="wide")

# Load model (cached)
@st.cache_resource
def load_model(model_path):
    return model_builder.load_xgboost_model(model_path)

@st.cache_data
def load_test_data():
    return pd.read_csv(FOMO_TEST_FILE)

@st.cache_data
def get_model_files():
    """Get list of all model files (cached)."""
    model_files = []
    if os.path.exists(MODEL_DIR):
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(('.json', '.pkl'))]
    return model_files

@st.cache_data
def detect_fomo_behavior(_model, test_data):
    """Detect FOMO behavior in all trading windows (cached per model)."""
    # Ensure feature columns are numeric
    result = test_data.copy()
    for col in FEATURE_COLS:
        result[col] = pd.to_numeric(result[col], errors='coerce')
    
    X_test = result[FEATURE_COLS]
    fomo_probabilities = _model.predict_proba(X_test)[:, 1]
    result['fomo_score'] = fomo_probabilities
    return result

@st.cache_data
def create_investor_table(test_data_with_scores):
    """Create investor summary table with FOMO levels (cached)."""
    # Create a lookup dictionary for fast access
    investor_data = []
    seen_investors = set()

    for idx, row in test_data_with_scores.iterrows():
        inv_id = row['investor_id']
        if inv_id not in seen_investors:
            score = row['fomo_score']
            level = fomo_level(score)
            level_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(level, "⚪")
            
            investor_data.append({
                'Investor ID': inv_id,
                'FOMO': f"{level_emoji} {level}",
                'FOMO Score': score,
                'Level': level  # For sorting
            })
            seen_investors.add(inv_id)
    
    # Create DataFrame and sort by score (highest first)
    df = pd.DataFrame(investor_data)
    df = df.sort_values('FOMO Score', ascending=False).reset_index(drop=True)
    
    return df

# Main app
st.title("📊 FOMO Investor Detection Dashboard")

# Sidebar
st.sidebar.header("Settings")

# Get list of all model files (cached - runs once)
model_files = get_model_files()
if not model_files:
    st.error(f"No model files found in {MODEL_DIR}")
    st.stop()

# Sort model files (newest first)
sorted_model_files = sorted(model_files, reverse=True)

# Model selector - show all files in selectbox
selected_model_file = st.sidebar.selectbox("Classifier Model", sorted_model_files)

# Load the selected model (cached per model file)
model = load_model(f"{MODEL_DIR}/{selected_model_file}")

# Load test data (cached - runs once)
test_data = load_test_data()

# Detect FOMO behavior in all windows (cached per model - runs once per model)
test_data_with_scores = detect_fomo_behavior(model, test_data)

# Create investor table (cached - runs once per model)
investor_table = create_investor_table(test_data_with_scores)

# Sidebar - Investor Selection
st.sidebar.subheader("Select Investor")

# Filter by FOMO Level
fomo_level_filter = st.sidebar.multiselect(
    "Filter by FOMO Level",
    options=["Low", "Medium", "High"],
    default=[],
    help=f"Filter by FOMO level: "
        + f"🟢 Low (<{th_fomo_score.LOW_FOMO*100:.1f}%), "
        + f"🟡 Medium ({th_fomo_score.LOW_FOMO*100:.1f}-{th_fomo_score.MEDIUM_FOMO*100:.1f}%), "
        + f"🔴 High (>{th_fomo_score.MEDIUM_FOMO*100:.1f}%). "
        + f"FOMO level shown in the table is from first trading window encountered per investor"
)

# Filter table based on FOMO level
if fomo_level_filter:
    filtered_table = investor_table[investor_table['Level'].isin(fomo_level_filter)].reset_index(drop=True)
else:
    filtered_table = investor_table.copy()

# Add 1-based index for display
display_table = filtered_table[['Investor ID', 'FOMO']].copy()
display_table.index = display_table.index + 1

# Display the scrollable table (read-only for viewing)
st.sidebar.dataframe(
    display_table,
    height=500,
    width='stretch',
    hide_index=False
)

# Selection with navigation
available_ids = filtered_table['Investor ID'].tolist()

# Track filter changes to reset selection
if 'previous_filter' not in st.session_state:
    st.session_state.previous_filter = fomo_level_filter
    
if st.session_state.previous_filter != fomo_level_filter:
    st.session_state.previous_filter = fomo_level_filter
    st.session_state.current_investor_index = 0  # Reset to first investor

# Initialize session state for current index
if 'current_investor_index' not in st.session_state:
    st.session_state.current_investor_index = 0

# Ensure index is within bounds
st.session_state.current_investor_index = max(0, min(st.session_state.current_investor_index, len(available_ids) - 1))

# Get current investor ID based on index
current_investor_id = str(available_ids[st.session_state.current_investor_index]) if available_ids else ""

# Text input with current investor ID (use both filter and index in key to handle both scenarios)
# Display row number starting from 1
filter_hash = str(sorted(fomo_level_filter))  # Create a hash of the filter to detect changes
selected_investor_id = st.sidebar.text_input(
    f"Investor ID (Row {st.session_state.current_investor_index + 1} of {len(available_ids)})",
    value=current_investor_id,
    placeholder="Type or paste Investor ID here...",
    key=f"investor_id_input_{filter_hash}_{st.session_state.current_investor_index}"
)

# Navigation buttons below text box (thinner)
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("⬅️ Previous", width='stretch', key="prev_btn"):
        st.session_state.current_investor_index = max(0, st.session_state.current_investor_index - 1)
        st.rerun()

with col2:
    if st.button("Next ➡️", width='stretch', key="next_btn"):
        st.session_state.current_investor_index = min(len(available_ids) - 1, st.session_state.current_investor_index + 1)
        st.rerun()

# Update index if user manually enters an ID
if selected_investor_id in available_ids:
    st.session_state.current_investor_index = available_ids.index(selected_investor_id)
    investor_id = selected_investor_id
else:
    if selected_investor_id:  # Only show warning if user entered something
        st.sidebar.warning(f"Investor ID '{selected_investor_id}' not found. Showing row {st.session_state.current_investor_index}.")
    investor_id = available_ids[st.session_state.current_investor_index] if available_ids else None

# === INDIVIDUAL INVESTOR ANALYSIS ===
st.header(f"🔍 Investor #{investor_id} - FOMO Behavior Analysis")

# Get ALL data for this investor (multiple trading windows)
investor_all_data = test_data_with_scores[test_data_with_scores['investor_id'] == investor_id]

# Show how many windows were analyzed for this investor
st.write(f"📊 **{len(investor_all_data)} trading window(s)** analyzed for this investor")

# If multiple windows, let user select which one to examine in detail
if len(investor_all_data) > 1:
    # Check if window_start exists
    has_dates = 'window_start' in investor_all_data.columns
    
    window_index = st.selectbox(
        "Select trading window to analyze:",
        range(len(investor_all_data)),
        format_func=lambda x: (
            f"Window #{x+1} - {investor_all_data.iloc[x]['window_start']} (FOMO Score: {investor_all_data.iloc[x]['fomo_score']:.4f})"
            if has_dates else
            f"Window #{x+1} (FOMO Score: {investor_all_data.iloc[x]['fomo_score']:.4f})"
        )
    )
    investor_data = investor_all_data.iloc[window_index]
    
    # Show summary table of all windows
    with st.expander("📋 View FOMO detection across all trading windows"):
        summary_cols = (['window_start'] if has_dates else []) + ['fomo_score'] + FEATURE_COLS
        windows_summary = investor_all_data[summary_cols].copy()
        windows_summary.insert(0, 'Window #', range(1, len(investor_all_data) + 1))
        windows_summary['FOMO Level'] = windows_summary['fomo_score'].apply(fomo_level)
        st.dataframe(windows_summary, width='stretch', hide_index=True)
else:
    investor_data = investor_all_data.iloc[0]

investor_score = investor_data['fomo_score']
investor_fomo_level = fomo_level(investor_score)

# Calculate key behavioral signals (top 3 features with highest absolute Impact Scores)
X_investor = investor_data[FEATURE_COLS].to_frame().T
# Ensure numeric dtypes for XGBoost
for col in FEATURE_COLS:
    X_investor[col] = pd.to_numeric(X_investor[col], errors='coerce')
shap_values = model_builder.get_shap_values(model, X_investor)

shap_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Impact Score': shap_values[0],
    'Feature Value': X_investor.iloc[0].values
})
shap_df = shap_df.sort_values('Impact Score', key=abs, ascending=False)
key_signals = shap_df.head(3)

# Display key metrics in cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="FOMO Score",
        value=f"{investor_score:.4f}",
        delta=f"{(investor_score - test_data_with_scores['fomo_score'].mean()):.4f} vs average",
        help="FOMO detection score (0=no FOMO, 1=strong FOMO). "
            + "Delta (shown below the score) shows how much this score differs from the average across all windows. "
            + "Positive delta = above average FOMO detected."
    )

with col2:
    level_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    st.metric(
        label="FOMO Level",
        value=f"{level_color.get(investor_fomo_level, '⚪')} {investor_fomo_level}"
    )

with col3:
    model_certainty = abs(investor_score - 0.5) * 2
    st.metric(
        label="Model Certainty",
        value=f"{model_certainty:.2%}",
        help="How confident the model is in its classification (close to 0.5 = uncertain, far from 0.5 = certain)"
    )

# Key Behavioral Signals
st.subheader("🎯 Key Behavioral Signals Detected")
st.write("Top 3 features indicating FOMO behavior in this window:")

signal_col1, signal_col2, signal_col3 = st.columns(3)

for idx, (col, row) in enumerate(zip([signal_col1, signal_col2, signal_col3], key_signals.itertuples())):
    with col:
        impact = "⬆️ Increases FOMO" if row._2 > 0 else "⬇️ Decreases FOMO"
        value_display = "None" if pd.isna(row._3) else f"{row._3:.4f}"
        st.markdown(f"""
        **#{idx+1}: {row.Feature}**
        - Value: `{value_display}`
        - Impact: {impact}
        - Impact Score: `{row._2:.4f}`
        """)

# FOMO Gauge
st.subheader("📊 FOMO Detection Score")
col1, col2 = st.columns([1, 2])

with col1:
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = investor_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "FOMO Score"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, th_fomo_score.LOW_FOMO], 'color': "lightgreen"},
                {'range': [th_fomo_score.LOW_FOMO, th_fomo_score.MEDIUM_FOMO], 'color': "yellow"},
                {'range': [th_fomo_score.MEDIUM_FOMO, 1], 'color': "salmon"}
            ],
            # 'threshold': {
            #     'line': {'color': "red", 'width': 4},
            #     'thickness': 0.75,
            #     'value': 0.05
            # }
        }
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    # Feature values comparison
    st.write("**Behavioral Features vs Average**")
    
    features_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'This Window': [investor_data[col] for col in FEATURE_COLS],
        'Average': [test_data_with_scores[col].mean() for col in FEATURE_COLS]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='This Window', x=features_df['Feature'], y=features_df['This Window'], width=0.3))
    fig.add_trace(go.Bar(name='Average', x=features_df['Feature'], y=features_df['Average'], width=0.3))
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, width='stretch')

# SHAP explanation
st.subheader("🔬 Model Explanation")

fig = px.bar(shap_df, x='Impact Score', y='Feature', orientation='h',
             title="How Each Feature Contributed to FOMO Detection",
             color='Impact Score',
             color_continuous_scale=['blue', 'red'],
             hover_data=['Feature Value'])
st.plotly_chart(fig, width='stretch')

# Summary box
model_certainty = abs(investor_score - 0.5) * 2
window_info = f" in window starting {investor_data.get('window_start', 'N/A')}" if 'window_start' in investor_data else ""

# Build behavioral evidence with human-readable descriptions
behavioral_evidence = []
for col in FEATURE_COLS:
    value = investor_data[col]
    if pd.isna(value):
        behavioral_evidence.append(f"  - **{col}**: None")
    elif col == 'n_buys':
        behavioral_evidence.append(f"  - **{col}**: {int(value)} buys made")
    elif col == 'avg_return_before_buy':
        behavioral_evidence.append(f"  - **{col}**: {value:.4f} (bought after {value*100:.2f}% price change)")
    elif col == 'buy_after_spike_ratio':
        behavioral_evidence.append(f"  - **{col}**: {value:.4f} ({value*100:.1f}% of buys were after spikes)")
    elif col == 'avg_missed_return':
        behavioral_evidence.append(f"  - **{col}**: {value:.4f} (missed {value*100:.2f}% returns on non-trading days)")
    elif col == 'n_trades':
        behavioral_evidence.append(f"  - **{col}**: {int(value)} total trades")
    else:
        behavioral_evidence.append(f"  - **{col}**: {value:.4f}")
behavioral_evidence_text = "  \n".join(behavioral_evidence)

st.info(f"""
**Detection Summary:**
- **Investor ID:** {investor_id}{window_info}
- **FOMO Detected:** {investor_fomo_level} level (score: {investor_score:.4f})
- **Model Certainty:** {model_certainty:.2%}
- **Primary Behavioral Signal:** {key_signals.iloc[0]['Feature']} (Impact Score: {key_signals.iloc[0]['Impact Score']:.4f})
- **Rank:** #{(test_data_with_scores['fomo_score'] > investor_score).sum() + 1} out of {len(test_data_with_scores)} windows analyzed

**Behavioral Evidence:**  
{behavioral_evidence_text}
""")

# Top 10 FOMO trading windows
st.subheader("🔥 Top 10 Windows with Highest FOMO Detected")
top_fomo = test_data_with_scores.nlargest(10, 'fomo_score')[['investor_id', 'fomo_score'] + FEATURE_COLS]
if 'window_start' in test_data_with_scores.columns:
    top_fomo.insert(1, 'window_start', test_data_with_scores.nlargest(10, 'fomo_score')['window_start'].values)
st.dataframe(top_fomo, width='stretch', hide_index=True)
