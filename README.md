# Customer Segmentation Analysis

## Overview
This project performs customer segmentation analysis on the Mall Customers dataset. The goal is to cluster customers based on their **Annual Income** and **Spending Score** to identify distinct customer groups for targeted marketing strategies.

## Features
The script implements and compares three clustering algorithms:
1.  **K-Means Clustering**:
    -   Determines the optimal number of clusters using the Elbow Method.
    -   Visualizes clusters with centroids.
2.  **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
    -   Identifies clusters based on density and detects outliers.
3.  **Hierarchical (Agglomerative) Clustering**:
    -   Builds a hierarchy of clusters.

## Visualizations
-   Scatterplots of Annual Income vs. Spending Score.
-   Elbow Method plot for K-Means.
-   Cluster visualizations for each algorithm.

## Prerequisites
Ensure you have the following Python libraries installed:
-   pandas
-   numpy
-   seaborn
-   matplotlib
-   scikit-learn

## Usage
1.  Ensure the dataset `Mall_Customers.csv` is in the correct path.
    > Note: You may need to update the file path in `script_AnishAhuja.py` if the dataset is not found.
2.  Run the script:
    ```bash
    python script_AnishAhuja.py
    ```

## File Structure
-   `script_AnishAhuja.py`: Main Python script for analysis and clustering.
-   `Mall_Customers.csv`: Dataset used for analysis.
-   `report_AnishAhuja.pdf`: Project report.