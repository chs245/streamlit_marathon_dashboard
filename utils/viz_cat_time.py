import plotly.graph_objects as go
from plotly.express.colors import sequential
import pandas as pd
import numpy as np
import plotly.express as px

def plot_finish_times_ridge(filtered_data, categories):

    # Convert timedelta to minutes
    filtered_data['finish_minutes'] = filtered_data['42.195 KM'].dt.total_seconds() / 60

    # Filter categories to only include those present in the filtered_data
    filtered_categories = [category for category in categories if category in filtered_data['Category'].unique()]

    # Create new figure
    fig = go.Figure()

    # Calculate the overall min and max for consistent scaling
    overall_min = filtered_data['finish_minutes'].min()
    overall_max = filtered_data['finish_minutes'].max()

    # Set the bin width to 1 minute
    bin_width = 1
    bins = np.arange(overall_min, overall_max + bin_width, bin_width)

    # Calculate the category-wise maximum bar heights for custom spacing
    category_max_heights = []
    category_counts = {}
    total_runners = len(filtered_data)

    for category in filtered_categories:
        cat_data = filtered_data[filtered_data['Category'] == category]['finish_minutes']
        category_counts[category] = len(cat_data)
        counts, _ = np.histogram(cat_data, bins=bins)
        category_max_heights.append(max(counts))

    # Plot each category with vertical offset based on the previous category's maximum height
    category_positions = []
    cumulative_offset = 0

    # Plot bars, baselines, and median lines for each category
    for i, category in enumerate(filtered_categories):
        cat_data = filtered_data[filtered_data['Category'] == category]['finish_minutes']
        counts, _ = np.histogram(cat_data, bins=bins)
        offset = cumulative_offset
        category_positions.append((category, offset))
        
        # Update cumulative_offset based on the max height of the current category or a minimum of 20 pixels
        minimum_spacing = 20
        calculated_spacing = category_max_heights[i] * 1.1
        cumulative_offset += max(calculated_spacing, minimum_spacing)  # Ensure at least 20 pixels between categories

        # Plot baseline for the category
        fig.add_trace(
            go.Scatter(
                x=[overall_min, overall_max],
                y=[offset, offset],
                mode='lines',
                line=dict(color='lightgray', width=1, dash='dash'),
                showlegend=False,  # Hide legend for baseline
                hoverinfo='skip'
            )
        )

        # Plot bars for each bin
        for j in range(len(bins) - 1):
            bar_height = counts[j]
            if bar_height > 0:
                color_rgb = px.colors.sample_colorscale("Viridis_r", [j / len(bins)])[0]

                hover_text = (
                    f"{category}<br>"
                    f"Time: {int(bins[j] // 60):02d}:{int(bins[j] % 60):02d} - {int(bins[j + 1] // 60):02d}:{int(bins[j + 1] % 60):02d}<br>"
                    f"Count: {counts[j]:,}<br>"
                    f"Athletes: {category_counts[category]:,}"
                )

                fig.add_trace(
                    go.Bar(
                        x=[(bins[j] + bins[j + 1]) / 2], marker_line_width=0,
                        y=[bar_height],
                        width=bin_width * 1.0,
                        base=offset,
                        marker_color=color_rgb,
                        showlegend=False,  # Hide legend for bars
                        text=hover_text,
                        hoverinfo='text',
                        opacity=1.0
                    )
                )

        # Calculate and plot median line for the category
        median_time = cat_data.median()
        # Ensure the median line for the last category does not extend beyond its baseline
        next_offset = cumulative_offset if i < len(filtered_categories) - 1 else cumulative_offset

        # Show legend for the median line only once
        show_median_legend = i == 0

        # Plot median line
        fig.add_trace(
            go.Scatter(
                x=[median_time, median_time],
                y=[offset, next_offset],
                mode='lines',
                line=dict(color='red', width=2, dash='dot'),
                name="Median Time" if show_median_legend else None,
                showlegend=show_median_legend
            )
        )

        # Plot median time text label
        median_time_hours = int(median_time // 60)
        median_time_minutes = int(median_time % 60)
        median_label = f"{median_time_hours:02d}:{median_time_minutes:02d}"

        fig.add_trace(
            go.Scatter(
                x=[median_time + 0.5],  # Slightly to the right of the median line
                y=[next_offset - 5],  # Position at the top of the median line
                mode='text',
                text=[median_label],
                textposition='bottom right',
                showlegend=False
            )
        )

    # Update layout
    fig.update_layout(
        title="Amsterdam Marathon 2024 - Finish Times Distribution by Age Group",
        xaxis_title="Finish Time",
        yaxis_title=None,
        template='plotly_white',
        height=800,
        hovermode='closest',
        showlegend=True,  # Show legend
        legend=dict(
            x=0.01,
            y=0.75,
            bgcolor='rgba(255, 255, 255, 0.8)',  # Optional: background color with transparency
            bordercolor='black',
            borderwidth=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticktext=[f"{cat} ({category_counts[cat]:,})" for cat, _ in category_positions],
            tickvals=[pos for _, pos in category_positions],
            tickmode="array",
            tickangle=0,
            tickfont=dict(size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        margin=dict(r=100)  # Increase right margin for annotation box
    )

    # Add annotation box for explanation
    fig.add_annotation(
        text=(
            "Explanation:<br>"
            "'V' stands for Women,<br>"
            "'M' for Men,<br>"
            "The number is the age group,<br>"
            "The value in parentheses<br>"
            "is the number of athletes."
        ),
        xref="paper",
        yref="paper",
        x=0.01,  # Position it to the left of the y-axis
        y=0.95,
        showarrow=False,
        font=dict(size=12, color='black'),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=5,
        bgcolor="white",
        opacity=0.8
    )

    # Format x-axis time labels with 30-minute intervals starting from the fastest time
    tick_values = np.arange(overall_min - (overall_min % 30), overall_max + 30, 30)
    if overall_min not in tick_values:
        tick_values = np.insert(tick_values, 0, overall_min)
    tick_texts = [f"{int(m // 60):02d}:{int(m % 60):02d}" for m in tick_values]

    fig.update_xaxes(
        ticktext=tick_texts,
        tickvals=tick_values
    )

    return fig
