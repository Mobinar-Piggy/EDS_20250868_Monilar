# Performance Metrics of AI Inference: A Statistical Analysis of Confidence Calibration
**Course:** Computer Programming 1 | **Topic ID:** AIP-03

## Project Overview
This repository contains a Python-based data analytics pipeline designed to audit an AI hiring model for algorithmic bias and confidence calibration. The pipeline processes inference logs to determine if the model exhibits overconfidence or underconfidence across different demographic groups (specifically focusing on Software Engineering candidates).

## Repository Architecture
- `main.py`: Core data ingestion, cleaning, and NumPy statistical analysis pipeline.
- `requirements.txt`: Required Python dependencies.
- `data/`: Contains the original and cleaned telemetry datasets.
- `outputs/`: Contains the generated static graphs and interactive Plotly animations.

## How to Run the Pipeline
1. Clone this repository to your local machine.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Execute the main pipeline to generate the cleaned data and statistical outputs:
    python main.py

## Key Findings
The pipeline detected a severe "Imposter Syndrome" bias within the AI model. For Associate degree candidates, the AI achieved a 97.6% predictive accuracy but only assigned itself a 22.8% average confidence score (a -74.7% calibration gap), whereas it exhibited higher confidence (53.0%) for PhD candidates despite lower actual accuracy (90.9%).