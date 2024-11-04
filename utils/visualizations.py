import plotly.graph_objects as go
from plotly.express.colors import sequential
import pandas as pd
import numpy as np
import plotly.express as px



def plot_rank_progression(top_athletes, split_columns,category_filter, time_measure):
    fig = go.Figure()

    # Define rank columns and split distances
    rank_columns = [f'Global Rank {split}' for split in split_columns]
    split_distances = [float(split.replace(' KM', '').replace(',', '.')) for split in split_columns]

    # Calculate min and max rank for the y-axis range with padding
    min_rank = top_athletes[rank_columns].min().min()
    max_rank = top_athletes[rank_columns].max().max()
    rank_padding = 0.5

    # Set up colors and line widths
    colors = (sequential.Viridis * ((len(top_athletes) // len(sequential.Viridis)) + 1))[:len(top_athletes)]
    line_widths = [max(10 - rank + 1, 2) for rank in top_athletes['Global Rank 42.195 KM']]

    # Track positions for start and end labels to handle overlap
    start_label_positions, end_label_positions = {}, {}

    # Flag to add "Finish Time" label only once
    finish_time_label_added = False

    # Plot rank changes for each athlete
    for idx, (_, row) in enumerate(top_athletes.iloc[::-1].iterrows()):
        pace_first_5km = row['Pace 5 KM Split']
        pace_last_segment = row['Pace 42.195 KM Split']

        # Calculate pace difference
        if pd.notna(pace_first_5km) and pd.notna(pace_last_segment):
            pace_difference = pace_last_segment - pace_first_5km
            sign = "-" if pace_difference < 0 else "+"
            minutes = int(abs(pace_difference) // 1)
            seconds = int((abs(pace_difference) % 1) * 60)
            pace_diff_text = f"{sign}{minutes}:{seconds:02d} min/km"
        else:
            pace_diff_text = "N/A"

        athlete_name = f"#{int(row['Overall Finish Rank'])} {row['Athlete Name']} ({row['Country']}) {int(row['SecondsFromGunTime'].total_seconds())}s start delay"
        athlete_first_name_with_diff = row['Athlete Name'].split()[0].capitalize()
        formatted_finish_time = f"{row['Finish Time'].hour}:{row['Finish Time'].minute:02}:{row['Finish Time'].second:02}".lstrip('0')

        original_idx = len(top_athletes) - 1 - idx

        # Plot the main trace
        fig.add_trace(go.Scatter(
            x=split_distances,
            y=row[rank_columns],
            mode='lines+markers+text',
            name=athlete_name,
            line=dict(width=line_widths[original_idx], color=colors[idx]),
            marker=dict(symbol='circle', size=20, color='white', line=dict(color=colors[idx], width=2)),
            text=row[rank_columns].astype(int),
            textfont=dict(color=colors[idx], size=10),
            textposition='middle center',
            cliponaxis=False,
            legendrank=row['Overall Finish Rank'],
            hovertext=[
                f"Athlete: {row['Athlete Name']}<br>Country: {row['Country']}<br>Rank: {rank}<br>Seconds From Gun Time: {int(row['SecondsFromGunTime'].total_seconds())}s" +
                f"<br>Split Time: {row[split].components.hours}:{row[split].components.minutes:02}:{row[split].components.seconds:02}" +
                (f"<br>Pace: {int(row.get(f'Pace {split} Split', 0) // 1)}:{int((row.get(f'Pace {split} Split', 0) % 1) * 60):02d} min/km" if pd.notna(row.get(f'Pace {split} Split', None)) else "<br>Pace: N/A")
                for rank, split in zip(row[rank_columns], split_columns)
            ],
            hoverinfo="text"
        ))

        # Determine position offsets for labels to handle overlaps
        start_x = split_distances[0] - 1
        start_y = float(row[rank_columns].iloc[0])

        if start_y in start_label_positions:
            start_label_positions[start_y] += 1
            y_offset = start_label_positions[start_y] * 0.3
        else:
            start_label_positions[start_y] = 0
            y_offset = 0

        # Add start label (left side)
        fig.add_trace(go.Scatter(
            x=[start_x],
            y=[start_y + y_offset],
            mode='text',
            text=[athlete_first_name_with_diff],
            textposition='middle left',
            textfont=dict(color=colors[idx]),
            showlegend=False,
            hoverinfo='skip',
            cliponaxis=False
        ))

        end_x = split_distances[-1] + 1
        end_y = float(row[rank_columns].iloc[-1])

        if end_y in end_label_positions:
            end_label_positions[end_y] += 1
            y_offset = end_label_positions[end_y] * 0.3
        else:
            end_label_positions[end_y] = 0
            y_offset = 0

        # Add end label (right side)
        fig.add_trace(go.Scatter(
            x=[end_x],
            y=[end_y + y_offset],
            mode='text',
            text=[formatted_finish_time],
            textposition='middle right',
            textfont=dict(color=colors[idx]),
            showlegend=False,
            hoverinfo='skip',
            cliponaxis=False
        ))

        # Add "Finish Time" label above the right label text for the first entry only
        if not finish_time_label_added:
            fig.add_trace(go.Scatter(
                x=[end_x],
                y=[end_y + y_offset - 0.5],  # Position slightly above the finish time label
                mode='text',
                text=time_measure,
                textposition='top right',
                textfont=dict(color='#4B0082', size=12),  # Dark purple color for font
                showlegend=False,
                hoverinfo='skip',
                cliponaxis=False
            ))
            finish_time_label_added = True  # Set flag to ensure label is only added once

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Rank Progression of Top {len(top_athletes)} Finish Athletes by {category_filter} Rank based on {time_measure} from Start",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title="Distance (KM)",
            range=[-3, 44.5],
            tickvals=split_distances,
            ticktext=split_columns,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Rank",
            range=[max_rank + rank_padding * 2, min_rank - rank_padding * 2],
            dtick=1,
            showgrid=True,
            showticklabels=False
        ),
        width=1200,
        height=800,
        plot_bgcolor='white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.1,
            traceorder="normal"
        ),
        margin=dict(l=100)
    )

    return fig
