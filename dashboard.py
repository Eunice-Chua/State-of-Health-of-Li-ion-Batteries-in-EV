import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

# Page Configuration
st.set_page_config(
    page_title="EV Battery Health Dashboard",
    page_icon="🔋",
    layout="wide"
)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'./data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load XGBoost Model
@st.cache_resource
def load_model():
    try:
        model_path = r'./best_xgb_model.json'
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize
df = load_data()
model = load_model()

# Sidebar Navigation
st.sidebar.title("🔋 EV Battery Monitor")
page = st.sidebar.selectbox("Navigation", [
    "Dashboard Overview",
    "Battery Analysis",
    "SOH Prediction",
])

# Dashboard Overview
if page == "Dashboard Overview":
    st.title("📊 Dashboard Overview")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_predictions = len(st.session_state.predictions) if 'predictions' in st.session_state else 0
        st.metric("Total Predictions", total_predictions)

    # Batteries in Good Health
    if 'predictions' in st.session_state and st.session_state.predictions:
        # Prepare data for battery distribution
        predictions_df = pd.DataFrame(st.session_state.predictions)
        good_health_count = len(predictions_df[predictions_df['condition'] == "Good health. ✅"])
        no_longer_usable_count = len(predictions_df[predictions_df['condition'] == "No longer usable. ⚠️"])
    else:
        good_health_count = 0
        no_longer_usable_count = 0

    with col2:
        st.metric("Batteries in Good Health", good_health_count)
    with col3:
        st.metric("No Longer Usable Batteries", no_longer_usable_count)

    # Overview Plot
    st.subheader("Battery Health Overview")
    fig = px.scatter(df, x='Cycle', y='Capacity', 
                    color='Battery', 
                    title='Battery Capacity vs Cycles')
    st.plotly_chart(fig, use_container_width=True)

# Battery Analysis
elif page == "Battery Analysis":
    # Load the specific data for Battery Analysis
    @st.cache_data
    def load_analysis_data():
        try:
            df_analysis = pd.read_csv(r'./combined_discharge_data.csv')

            return df_analysis
        except Exception as e:
            st.error(f"Error loading analysis data: {e}")
            return None

    st.title("📈 Battery Analysis")
    
    # Load the dataset
    df_analysis = load_analysis_data()
    
    if df_analysis is not None:
        # User selects batteries
        batteries = st.multiselect("Select Batteries", df_analysis['Battery'].unique())
        
        # Predefined cycles for selection
        available_cycles = [1, 30, 60, 90, 120, 150]
        selected_cycles = st.multiselect("Select Cycles for Visualization", available_cycles)

        # User selects metrics
        metrics = st.multiselect("Select Metrics", 
                                 ['Capacity', 'Temperature_measured', 'Voltage_measured'])
        
        if batteries and metrics and selected_cycles:
            for metric in metrics:
                # Different x-axes depending on the metric
                if metric == "Capacity":
                    x_axis = "Cycle"
                    filtered_data = df_analysis[df_analysis['Battery'].isin(batteries)]
                else:
                    x_axis = "Time"
                    filtered_data = df_analysis[
                        (df_analysis['Battery'].isin(batteries)) & (df_analysis['Cycle'].isin(selected_cycles))
                    ]

                # Generate the plot
                fig = px.line(
                    filtered_data,
                    x=x_axis,
                    y=metric,
                    color="Battery",
                    line_group="Cycle" if metric != "Capacity" else None,
                    title=f"{metric} Analysis ({x_axis} as X-axis, Selected Cycles: {', '.join(map(str, selected_cycles))})",
                    labels={"Battery": "Battery", x_axis: x_axis, metric: metric}
                )
                
                # Update line style and colors
                fig.update_traces(mode="lines")
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(filtered_data[metrics + ['Battery', 'Cycle']].describe())
            
            # Export filtered data as CSV
            csv_data = filtered_data.to_csv(index=False).encode('utf-8')  # Encode CSV data in UTF-8
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f'battery_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )
        else:
            st.info("Please select batteries, metrics, and cycles to generate the visualization.")
    else:
        st.warning("No data available for analysis.")

# SOH Prediction
elif page == "SOH Prediction":
    st.title("🤖 SOH Prediction")
    
    # Create 3 tabs
    pred_tab, history_tab, delete_tab = st.tabs(["🪄 Make Prediction", "📜 Prediction History", "🗑️ Delete Record"])
    
    with pred_tab:
        st.subheader("Enter Battery Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            cycle = st.number_input("Cycle", 
                                    value=0, 
                                    step=1, 
                                    min_value=0)
            voltage_mean = st.number_input("Voltage Mean (V)", 
                                           value=0.0000, 
                                           step=0.0001, 
                                           format="%.4f", 
                                           min_value=0.0000)
            temperature_max = st.number_input("Maximum Temperature (°C)", 
                                              value=0.0, 
                                              step=0.0001, 
                                              format="%.4f", 
                                              min_value=0.0)
        with col2:
            time_at_max_temp = st.number_input("Time at Max Temperature (secs)", 
                                               value=0.0, 
                                               step=0.0001, 
                                               format="%.4f", 
                                               min_value=0.0)
            discharge_duration = st.number_input("Discharge Duration (secs)", 
                                                 value=0.0, 
                                                 step=0.0001, 
                                                 format="%.4f", 
                                                 min_value=0.0)
        
        # Make Prediction
        if st.button("Predict SOH"):
            try:
                # Prepare input data
                input_data = pd.DataFrame([[cycle, voltage_mean, temperature_max, time_at_max_temp, discharge_duration]], 
                                          columns=['Cycle', 'Voltage_mean', 'Temperature_max', 'Time_at_max_temp', 'Discharge_duration'])
                dmatrix = xgb.DMatrix(input_data)
                prediction = model.predict(dmatrix)[0]
                soh_percentage = round(prediction * 100, 2)

                # Determine the battery condition based on SOH
                if soh_percentage >= 80.0:
                    condition = "Good health. ✅"
                    st.success(f"Predicted SOH: {soh_percentage:.2f}% - {condition}")
                else:
                    condition = "No longer usable. ⚠️"
                    st.error(f"Predicted SOH: {soh_percentage:.2f}% - {condition}")
        
                # Store prediction in session state
                if 'predictions' not in st.session_state:
                    st.session_state.predictions = []
                
                # Add timestamp and prediction data
                prediction_data = {
                    'timestamp': datetime.now(),
                    'cycle': cycle,
                    'voltage_mean': voltage_mean,
                    'temperature_max': temperature_max,
                    'time_at_max_temp': time_at_max_temp,
                    'discharge_duration': discharge_duration,
                    'predicted_soh (%)': soh_percentage,
                    'condition': condition
                }
                st.session_state.predictions.append(prediction_data)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    with history_tab:
        
        if 'predictions' in st.session_state and st.session_state.predictions:
            # Convert predictions to DataFrame
            history_df = pd.DataFrame(st.session_state.predictions)
            history_df.reset_index(inplace=True, drop=False)  # Add the original index as a column
            history_df['index'] = history_df['index'] + 1  # Adjust index to start from 1
            
            # Add a filter for battery condition
            st.subheader("Filter Prediction History")
            condition_filter = st.selectbox(
                "Select Battery Condition to Filter",
                options=["All", "Good health. ✅", "No longer usable. ⚠️"],
                index=0  # Default to "All"
            )

            # Apply the filter
            if condition_filter != "All":
                filtered_df = history_df[history_df['condition'] == condition_filter]
            else:
                filtered_df = history_df

            # # Display predictions table
            st.subheader("Predictions Table")
            st.dataframe(filtered_df)
            
            # Download option
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'soh_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )
            
            # Visualization
            st.subheader("Prediction Trends")

            # Scatter plot of SOH vs Cycle
            fig1 = px.scatter(history_df, x='cycle', y='predicted_soh (%)',
                                title='SOH vs Cycle',
                                labels={'cycle': 'Cycle', 'predicted_soh (%)': 'SOH (%)'})
            st.plotly_chart(fig1, use_container_width=True)

            # Scatter plot of SOH vs Voltage Mean
            fig2 = px.scatter(history_df, x='voltage_mean', y='predicted_soh (%)',
                  title='SOH vs Voltage Mean',
                  labels={'voltage_mean': 'Voltage Mean (V)', 'predicted_soh (%)': 'SOH (%)'})
            st.plotly_chart(fig2, use_container_width=True)

            # Scatter plot of SOH vs Maximum Temperature
            fig3 = px.scatter(history_df, x='temperature_max', y='predicted_soh (%)',
                  title='SOH vs Maximum Temperature',
                  labels={'temperature_max': 'Maximum Temperature (°C)', 'predicted_soh (%)': 'SOH (%)'})
            st.plotly_chart(fig3, use_container_width=True)

            # Scatter plot of SOH vs Time at Max Temperature
            fig4 = px.scatter(history_df, x='time_at_max_temp', y='predicted_soh (%)',
                  title='SOH vs Time at Max Temperature',
                  labels={'time_at_max_temp': 'Time at Max Temperature (secs)', 'predicted_soh (%)': 'SOH (%)'})
            st.plotly_chart(fig4, use_container_width=True)

            # Scatter plot of SOH vs Discharge Duration
            fig5 = px.scatter(history_df, x='discharge_duration', y='predicted_soh (%)',
                  title='SOH vs Discharge Duration',
                  labels={'discharge_duration': 'Discharge Duration (secs)', 'predicted_soh (%)': 'SOH (%)'})
            st.plotly_chart(fig5, use_container_width=True)

            # Correlation matrix for parameters
            correlation_data = history_df[['cycle', 'voltage_mean', 'temperature_max', 'time_at_max_temp', 'discharge_duration', 'predicted_soh (%)']]
            correlation_matrix = correlation_data.corr()

            # Plot heatmap
            fig5, ax = plt.subplots()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
            st.pyplot(fig5)

        else:
            st.info("No prediction history available yet. Make some predictions to see them here!")
    
    with delete_tab:
        if 'predictions' in st.session_state and st.session_state.predictions:
            history_df = pd.DataFrame(st.session_state.predictions)
        
            # Add index for row selection
            history_df.reset_index(inplace=True, drop=False)  # Ensure an index column exists

            st.subheader("Select Rows to Delete")
        
            rows_to_delete = st.multiselect(
                "Select rows to delete",
                options=history_df.index,
                format_func=lambda x: f"Row {x+1}: {history_df.loc[x, 'timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - SOH: {history_df.loc[x, 'predicted_soh (%)']:.4f}"
        )
        
            if rows_to_delete and st.button("Delete Selected Rows"):
                st.session_state.predictions = [pred for i, pred in enumerate(st.session_state.predictions) if i not in rows_to_delete]
                st.success("Selected rows deleted successfully!")
                st.experimental_set_query_params() # This triggers a rerun in Streamlit
                # st.experimental_rerun()

            # Clear history option
            if st.button("Clear All History"):
                st.session_state.predictions = []
                st.experimental_set_query_params()
        else:
            st.info("No predictions to delete.")
