import streamlit as st
from utils.data_loader import load_data
from utils.filters import filter_data
from utils.rank_calculator import calculate_ranks
from utils.visualizations import plot_rank_progression
from utils.viz_cat_time import plot_finish_times_ridge

# Set up Streamlit layout and configuration
st.set_page_config(layout="wide")

# Load data
athlete_data, data_types = load_data()
# Convert split columns to timedelta for time calculations
split_columns = ['5 KM', '10 KM', '15 KM', '20 KM', '25 KM', '30 KM', '35 KM', '40 KM', '42.195 KM']

if athlete_data is not None:
    data = athlete_data.copy()
        
    # Define custom sorting order for categories
    category_sort_order = {
        'Msen': 30,  # Treat 'Msen' as 'M30' in sorting
        'M35': 35,
        'M40': 40,
        'M45': 45,
        'M50': 50,
        'M55': 55,
        'M60': 60,
        'M65': 65,
        'M70': 70,
        'M75': 75,
        'Vsen': 30,  # Treat 'Vsen' as 'V30' in sorting
        'V35': 35,
        'V40': 40,
        'V45': 45,
        'V50': 50,
        'V55': 55,
        'V60': 60,
        'V65': 65,
        'V70': 70,
        'V75': 75
    }

    # Filter categories to only include those that start with 'M' or 'V'
    filtered_categories = [
        category for category in data['Category'].unique().tolist()
        if category.startswith('M') or category.startswith('V')
    ]

    # Ensure categories are unique and sort them using the custom order
    categories = sorted(
        filtered_categories,
        key=lambda x: category_sort_order.get(x, float('inf'))  # Fallback for missing keys
    )


    # Define the parameters for filtering and displaying data with Streamlit widgets
    with st.sidebar:
        category_filter = st.selectbox("Select Category Filter:", options=['Global'] + categories)
        gender_filter = st.radio("Select Gender Filter:", options=['All', 'Male', 'Female'], index=2)
        time_measure = st.selectbox("Select Time Measure:", options=['Chip Time', 'Gun Time'])
        top_n = st.slider("Select Number of Top Athletes to Display:", min_value=1, max_value=100, value=10)


    # Apply filters
    filtered_data = filter_data(data, category_filter, gender_filter)

    # Calculate ranks
    top_athletes, aggregated_data = calculate_ranks(filtered_data, time_measure, top_n)
    
    # Display filtered data
    if not top_athletes.empty:
        st.write("Top Athletes Data")
        st.dataframe(top_athletes)
    else:
        st.write("No top athletes data available for the selected filters.")

    # Display aggregated data
    st.write("Aggregated Data by Category")
    st.dataframe(aggregated_data)

    # Visualization: Rank Progression of Top Athletes
    st.write("Rank Progression of Top Athletes")
    rank_progression_fig = plot_rank_progression(top_athletes, split_columns, category_filter, time_measure)
    st.plotly_chart(rank_progression_fig)

    # Plot finish times ridge plot
    st.write("Finish Times Ridge Plot (Bar Version)")
    ridge_plot_fig = plot_finish_times_ridge(filtered_data,categories)
    st.plotly_chart(ridge_plot_fig)
