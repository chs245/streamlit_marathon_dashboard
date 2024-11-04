def filter_data(data, category_filter, gender_filter):
    filtered_data = data.copy()

    if category_filter != 'Global':
        filtered_data = filtered_data[filtered_data['Category'] == category_filter]

    if gender_filter == 'Male':
        filtered_data = filtered_data[filtered_data['Category'].str.startswith('M')]
    elif gender_filter == 'Female':
        filtered_data = filtered_data[filtered_data['Category'].str.startswith('V')]

    return filtered_data
