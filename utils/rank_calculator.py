def calculate_ranks(filtered_data, time_measure, top_n):
    split_columns = ['5 KM', '10 KM', '15 KM', '20 KM', '25 KM', '30 KM', '35 KM', '40 KM', '42.195 KM']
    
    if time_measure == 'Chip Time':
        filtered_data['Overall Finish Rank'] = filtered_data['Chip Time'].rank(method='min').astype('Int64')
        filtered_data['Finish Time'] = filtered_data['Chip Time']
        for split in split_columns:
            rank_col = f'Global Rank {split}'
            filtered_data[rank_col] = filtered_data[split].rank(method='min').astype('Int64')
        top_athletes = filtered_data.sort_values(by='Chip Time').head(top_n).reset_index(drop=True)
    elif time_measure == 'Gun Time':
        filtered_data['Overall Finish Rank'] = filtered_data['Gun Time'].rank(method='min').astype('Int64')
        filtered_data['Finish Time'] = filtered_data['Gun Time']
        for split in split_columns:
            adjusted_split_col = f'Adjusted {split}'
            filtered_data[adjusted_split_col] = filtered_data[split] + filtered_data['SecondsFromGunTime']
            rank_col = f'Global Rank {split}'
            filtered_data[rank_col] = filtered_data[adjusted_split_col].rank(method='min').astype('Int64')
        top_athletes = filtered_data.sort_values(by='Gun Time').head(top_n).reset_index(drop=True)

    agg_columns = split_columns + ['Chip Time', 'Gun Time']
    aggregated_data = (
        filtered_data.groupby('Category')[agg_columns]
        .agg(['mean', 'count', 'min'])
        .reset_index()
    )

    return top_athletes, aggregated_data
