import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Function to compute the weighted rank
def compute_weighted_rank(df, weights, invert_cols, rank_range):
    for col in invert_cols:
        df[col] = df[col].max() - df[col]
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    weighted_rank = (scaled_df * weights).sum(axis=1)
    
    # Scale the weighted rank to the specified range
    rank_scaler = MinMaxScaler(feature_range=rank_range)
    scaled_weighted_rank = rank_scaler.fit_transform(weighted_rank.values.reshape(-1, 1)).flatten()
    
    return scaled_weighted_rank

# Streamlit App
st.title("Custom Weighted Rank Calculator")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Columns in the dataset:")
    st.write(df.columns.tolist())

    # Select columns to include in the ranking
    selected_columns = st.multiselect("Select columns to include in the ranking", df.columns.tolist(), default=df.columns.tolist())
    if selected_columns:
        selected_df = df[selected_columns]
        st.write("### Selected columns:")
        st.write(selected_columns)

        # Check for non-numeric data and NaNs
        non_numeric_cols = [col for col in selected_columns if not pd.api.types.is_numeric_dtype(df[col])]
        nan_cols = [col for col in selected_columns if df[col].isna().any()]

        if non_numeric_cols or nan_cols:
            if non_numeric_cols:
                st.error(f"Columns must be formatted as numbers or decimals. Non-numeric columns detected: {', '.join(non_numeric_cols)}")
            if nan_cols:
                st.error(f"Columns contain NaNs: {', '.join(nan_cols)}. Please handle missing data before proceeding.")
        else:
            # Assign weights to the selected columns
            st.write("### Assign weights to the selected columns (0-100):")
            weights = []
            for col in selected_columns:
                weight = st.slider(f"Weight for {col}", 0, 100, 0)
                weights.append(weight / 100.0)  # Convert to a fraction of 1
            
            # Normalize weights so that they sum up to 1
            total_weight = sum(weights)
            if total_weight != 1:
                st.error("Total weight must sum up to 100%.")
            else:
                weight_dict = dict(zip(selected_columns, weights))
                st.write("Normalized Weights:")
                st.write(weight_dict)

                # Select columns to be inverted with a descriptive label
                invert_cols = st.multiselect("Which columns should be higher the metric the worse the score? Select columns to be inverted (if any)", selected_columns)
                
                # Set the range for the weighted rank
                st.write("### Set the range for the weighted rank:")
                rank_min = st.slider("Minimum rank", 1, 100, 1)
                rank_max = st.slider("Maximum rank", 1, 100, 100)
                if rank_min >= rank_max:
                    st.error("Minimum rank must be less than maximum rank.")
                else:
                    rank_range = (rank_min, rank_max)

                    # Calculate the weighted rank
                    if st.button("Calculate Weighted Rank"):
                        weighted_rank = compute_weighted_rank(selected_df, weights, invert_cols, rank_range)
                        df['Weighted Rank'] = weighted_rank

                        # Sort the DataFrame by Weighted Rank in descending order
                        df_sorted = df.sort_values(by='Weighted Rank', ascending=False).reset_index(drop=True)
                        
                        # Add Rank column
                        df_sorted['Rank'] = range(1, len(df_sorted) + 1)

                        # Reorder columns
                        cols = df_sorted.columns.tolist()
                        cols.insert(0, cols.pop(cols.index('Rank')))  # Move Rank to the first column
                        cols.insert(2, cols.pop(cols.index('Weighted Rank')))  # Move Weighted Rank to the third column
                        df_sorted = df_sorted[cols]
                        
                        st.write("### Data with Weighted Rank and Rank (Sorted):")
                        st.write(df_sorted)
