import pandas as pd
import numpy as np
import os

# ==========================================
# FUNCTION 1: DATA INGESTION & ERROR HANDLING
# ==========================================
def ingest_data(filepath):
    """Loads the dataset safely."""
    try:
        print(f"[*] Ingesting telemetry from: {filepath}")
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print("[!] CRITICAL ERROR: Dataset not found.")
        return None
    except Exception as e:
        print(f"[!] CRITICAL ERROR: {e}")
        return None

# ==========================================
# FUNCTION 2: CLEANING & UNIQUE FILTERING
# ==========================================
def clean_and_filter(df):
    """Handles nulls, duplicates, and applies the unique project filter."""
    try:
        # 1. Clean: Drop nulls and duplicates
        initial_len = len(df)
        df = df.dropna().drop_duplicates()
        
        # 2. Type Correction: Ensure scores are floats
        df['AI_Score'] = df['AI_Score'].astype(float)
        
        print(f"[*] Cleaning complete. Removed {initial_len - len(df)} corrupted rows.")

        # 3. Unique Filter: Isolate 'Software Engineer' roles for mathematical uniqueness
        df_filtered = df[df['Job_Category'] == 'Software Engineer'].copy()
        print(f"[*] Unique Filter Applied: {len(df_filtered)} Software Engineer records isolated.")
        
        # Save output
        os.makedirs('data', exist_ok=True)
        df_filtered.to_csv('data/dataset_cleaned.csv', index=False)
        return df_filtered
    except KeyError as e:
        print(f"[!] ERROR: Missing expected column: {e}")
        return None

# ==========================================
# FUNCTION 3: DESCRIPTIVE STATISTICS (STRICT NUMPY)
# ==========================================
def compute_descriptive_stats(df):
    """Computes core calibration metrics using NumPy."""
    print("\n--- DESCRIPTIVE STATISTICS (CONFIDENCE CALIBRATION) ---")
    ai_scores = np.array(df['AI_Score'])
    
    mean_val = np.mean(ai_scores)
    median_val = np.median(ai_scores)
    std_val = np.std(ai_scores)
    var_val = np.var(ai_scores)
    
    print(f"Mean AI Confidence:  {mean_val:.2f}%")
    print(f"Median AI Confidence:{median_val:.2f}%")
    print(f"Standard Deviation:  {std_val:.2f}")
    print(f"Variance:            {var_val:.2f}")
    
    # Skewness/Outlier logic
    if mean_val > median_val:
        print("Distribution Alert: Data is right-skewed (Model leans towards overconfidence).")
    else:
        print("Distribution Alert: Data is left-skewed (Model leans towards underconfidence).")

# ==========================================
# FUNCTION 4: CORRELATION ANALYSIS
# ==========================================
def compute_correlation(df):
    """Calculates relationship between Skill Fit and AI Confidence."""
    print("\n--- CORRELATION ANALYSIS ---")
    skill_scores = np.array(df['Skill_Fit_Score'])
    ai_scores = np.array(df['AI_Score'])
    
    # Use NumPy correlation coefficient matrix
    corr_matrix = np.corrcoef(skill_scores, ai_scores)
    r_value = corr_matrix[0, 1]
    
    print(f"Correlation (Skill vs. AI Confidence): {r_value:.4f}")
    if r_value > 0.7:
        print("Interpretation: Strong positive correlation. The AI accurately scales confidence with actual candidate skill.")
    else:
        print("Interpretation: Weak correlation. The AI's confidence does not align well with actual candidate skill.")

# ==========================================
# FUNCTION 5: COMPARATIVE ANALYSIS (BIAS)
# ==========================================
def evaluate_algorithmic_bias(df):
    """Compares model accuracy across different demographic groups."""
    print("\n--- COMPARATIVE ANALYSIS (ALGORITHMIC BIAS) ---")
    groups = df['Education_Level'].unique()
    
    for group in groups:
        subset = df[df['Education_Level'] == group]
        if len(subset) == 0: continue
        
        # Accuracy: Where AI_Decision matches Final_Decision
        ai_decisions = np.array(subset['AI_Decision'])
        ground_truth = np.array(subset['Final_Decision'])
        
        correct_preds = np.sum(ai_decisions == ground_truth)
        accuracy = (correct_preds / len(subset)) * 100
        
        # Average Confidence
        avg_conf = np.mean(np.array(subset['AI_Score']))
        calibration_gap = avg_conf - accuracy
        
        print(f"[{group}] Target Accuracy: {accuracy:.1f}% | Avg Model Confidence: {avg_conf:.1f}% | Calibration Gap: {calibration_gap:.1f}%")

# ==========================================
# MAIN ARCHITECTURE EXECUTION
# ==========================================
if __name__ == "__main__":
    filepath = 'data/dataset_original.csv'
    
    # Run the Pipeline
    raw_df = ingest_data(filepath)
    if raw_df is not None:
        clean_df = clean_and_filter(raw_df)
        if clean_df is not None:
            compute_descriptive_stats(clean_df)
            compute_correlation(clean_df)
            evaluate_algorithmic_bias(clean_df)
            print("\n[*] Phase 3 Complete. Pipeline executed successfully.")

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os

def create_static_visuals(df):
    """Generates and saves 3 static graphs."""
    print("[*] Generating Static Graphs...")
    os.makedirs('outputs', exist_ok=True)

    # 1. Histogram: Distribution of AI Confidence
    plt.figure(figsize=(8, 5))
    plt.hist(df['AI_Score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of AI Confidence Scores')
    plt.xlabel('AI Confidence Score (%)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('outputs/static_1_histogram.png')
    plt.close()

    # 2. Boxplot: Confidence by Education Level (The Bias Proof)
    plt.figure(figsize=(8, 5))
    df.boxplot(column='AI_Score', by='Education_Level', grid=False)
    plt.title('AI Confidence Spread by Education Level')
    plt.suptitle('') # Removes default pandas subtitle
    plt.xlabel('Education Level')
    plt.ylabel('AI Confidence Score (%)')
    plt.savefig('outputs/static_2_boxplot.png')
    plt.close()

    # 3. Scatter Plot: Skill vs AI Score
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Skill_Fit_Score'], df['AI_Score'], alpha=0.6, color='coral')
    plt.title('Skill Fit vs. AI Confidence Score')
    plt.xlabel('Actual Skill Fit Score')
    plt.ylabel('AI Confidence Score')
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/static_3_scatter.png')
    plt.close()

def create_animated_visuals(df):
    """Generates and saves 2 animated graphs using Plotly."""
    print("[*] Generating Animated Graphs...")

    # Sort data so the animation flows logically
    df = df.sort_values('Years_Experience')

    # 1. Animated Scatter: How Confidence scales with Skill over Years of Experience
    fig1 = px.scatter(
        df, 
        x="Skill_Fit_Score", 
        y="AI_Score", 
        animation_frame="Years_Experience", 
        color="Education_Level",
        title="Animation 1: AI Confidence Tracking Skill Across Experience Levels",
        range_x=[0, 100], 
        range_y=[0, 100]
    )
    fig1.write_html('outputs/anim_1_scatter.html')

    # 2. Animated Histogram: Confidence spread shifting across Experience Levels
    fig2 = px.histogram(
        df, 
        x="AI_Score", 
        animation_frame="Years_Experience", 
        color="Education_Level", 
        title="Animation 2: Shifting AI Confidence Distributions by Experience",
        range_x=[0, 100]
    )
    fig2.write_html('outputs/anim_2_histogram.html')

if __name__ == "__main__":
    filepath = 'data/dataset_cleaned.csv'
    
    try:
        clean_df = pd.read_csv(filepath)
        create_static_visuals(clean_df)
        create_animated_visuals(clean_df)
        print("[*] Phase 4 Complete. Check your 'outputs/' folder for 5 new files.")
    except FileNotFoundError:
        print("[!] ERROR: Run main.py first to generate data/dataset_cleaned.csv")