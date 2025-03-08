import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import calendar
from folium.plugins import HeatMap, MarkerCluster
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import plotly.io as pio
import os
pio.renderers.default = "browser"

# Set style for better visualizations
plt.style.use('seaborn-v0_8') 
sns.set_theme()
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Read the data
df = pd.read_csv('data/University_Dataset(tmc_raw_data_2020_2029).csv')

# Convert datetime columns
df['count_date'] = pd.to_datetime(df['count_date'])
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# Extract day of week (as a number 0-6, where 0 is Sunday)
df['day_of_week'] = df['count_date'].dt.dayofweek
# Adjust to make Sunday (6) the first day of the week (0)
df['day_of_week'] = (df['day_of_week'] + 1) % 7
# Convert to day names
day_map = {
    0: 'Sunday',
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
}
df['day_of_week'] = df['day_of_week'].map(day_map)

print("Unique day_of_week values:", df['day_of_week'].unique())

df['month'] = df['count_date'].dt.month
df['hour'] = df['start_time'].dt.hour

# Calculate total vehicles for each direction and type
direction_columns = {
    'cars': ['_appr_cars_r', '_appr_cars_t', '_appr_cars_l'],
    'trucks': ['_appr_truck_r', '_appr_truck_t', '_appr_truck_l'],
    'buses': ['_appr_bus_r', '_appr_bus_t', '_appr_bus_l'],
    'bikes': ['_appr_bike'],
    'peds': ['_appr_peds']
}

# Calculate totals by direction and vehicle type
for vehicle_type, suffixes in direction_columns.items():
    for direction in ['n', 's', 'e', 'w']:
        cols = [direction + suffix for suffix in suffixes]
        df[f'total_{direction}_{vehicle_type}'] = df[cols].sum(axis=1)

# Calculate total for each vehicle type
for vehicle_type in direction_columns.keys():
    df[f'total_{vehicle_type}'] = sum([df[f'total_{direction}_{vehicle_type}'] for direction in ['n', 's', 'e', 'w']])

# Calculate overall total traffic
df['total_traffic'] = df[[col for col in df.columns if col.startswith('total_') and '_' in col.split('total_')[1]]].sum(axis=1)

def create_heatmap_by_time():
    """Create a heatmap showing traffic patterns by hour and day of week"""
    # Create the pivot table
    pivot_table = df.pivot_table(
        values='total_traffic',
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )
    
    # Get the available days in the pivot table
    available_days = pivot_table.columns.tolist()
    print(f"Available days in the data: {available_days}")
    
    # Define the preferred order
    preferred_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    # Only use days that exist in both lists
    days_order = [day for day in preferred_order if day in available_days]
    
    # If we have days to reorder, do so
    if days_order:
        pivot_table = pivot_table[days_order]
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, fmt='.0f')
    plt.title('Traffic Patterns by Hour and Day of Week', fontsize=16)
    plt.ylabel('Hour of Day', fontsize=14)
    plt.xlabel('Day of Week', fontsize=14)
    plt.tight_layout()
    plt.savefig('time_heatmap.png', dpi=300)
    plt.close()

def create_vehicle_type_heatmaps():
    """Create separate heatmaps for each vehicle type"""
    vehicle_types = ['cars', 'trucks', 'buses', 'bikes', 'peds']
    
    # Define the preferred order
    preferred_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    for vehicle_type in vehicle_types:
        pivot_table = df.pivot_table(
            values=f'total_{vehicle_type}',
            index='hour',
            columns='day_of_week',
            aggfunc='mean'
        )
        
        # Get the available days in the pivot table
        available_days = pivot_table.columns.tolist()
        
        # Only use days that exist in both lists
        days_order = [day for day in preferred_order if day in available_days]
        
        # If we have days to reorder, do so
        if days_order:
            pivot_table = pivot_table[days_order]
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, fmt='.1f')
        plt.title(f'{vehicle_type.capitalize()} Traffic by Hour and Day of Week', fontsize=16)
        plt.ylabel('Hour of Day', fontsize=14)
        plt.xlabel('Day of Week', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{vehicle_type}_heatmap.png', dpi=300)
        plt.close()

def create_location_heatmap():
    """Create an interactive heatmap using Folium"""
    location_data = df.groupby(['latitude', 'longitude'])['total_traffic'].mean().reset_index()
    
    # Create base map
    m = folium.Map(location=[43.7001, -79.4163], zoom_start=11)
    
    # Add heatmap layer
    heat_data = [[row['latitude'], row['longitude'], row['total_traffic']] for index, row in location_data.iterrows()]
    HeatMap(heat_data).add_to(m)
    
    m.save('location_heatmap.html')

def create_top_locations_map():
    """Create a map showing the top 20 busiest locations with color-coded markers"""
    # Calculate average traffic for each location
    location_traffic = df.groupby(['location_name', 'latitude', 'longitude'])['total_traffic'].mean().reset_index()
    
    # Get top 20 busiest locations
    top_locations = location_traffic.sort_values('total_traffic', ascending=False).head(20)
    
    # Create map
    m = folium.Map(location=[43.7001, -79.4163], zoom_start=11)
    
    # Create a colormap (using the updated way)
    colormap = plt.colormaps['YlOrRd']
    
    # Normalize traffic values for coloring
    min_traffic = top_locations['total_traffic'].min()
    max_traffic = top_locations['total_traffic'].max()
    
    # Add markers for each top location
    for i, row in top_locations.iterrows():
        # Normalize the traffic value to get a color
        normalized_traffic = (row['total_traffic'] - min_traffic) / (max_traffic - min_traffic)
        color = '#%02x%02x%02x' % tuple(int(c * 255) for c in colormap(normalized_traffic)[:3])
        
        # Create a circle marker
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=15,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<strong>{row['location_name']}</strong><br>Average Traffic: {row['total_traffic']:.0f}"
        ).add_to(m)
        
        # Add a text label
        folium.map.Marker(
            [row['latitude'], row['longitude']],
            icon=folium.DivIcon(
                icon_size=(150,36),
                icon_anchor=(75,0),
                html=f'<div style="font-size: 10pt; color: black; font-weight: bold; text-align: center;">{i+1}</div>'
            )
        ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white; padding: 10px;
                ">
      <p><strong>Traffic Intensity</strong></p>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #ffffb2; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Lower</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #fecc5c; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Medium</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #f03b20; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Higher</span>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save('top_locations_map.html')

def create_vehicle_distribution_by_location():
    """Create a stacked bar chart showing vehicle distribution at top 10 locations"""
    # Get total counts by vehicle type and location
    location_vehicle_counts = df.groupby('location_name').agg({
        'total_cars': 'mean',
        'total_trucks': 'mean',
        'total_buses': 'mean',
        'total_bikes': 'mean',
        'total_peds': 'mean'
    }).reset_index()
    
    # Add total column
    location_vehicle_counts['total'] = location_vehicle_counts[['total_cars', 'total_trucks', 
                                                              'total_buses', 'total_bikes', 
                                                              'total_peds']].sum(axis=1)
    
    # Get top 10 locations
    top_locations = location_vehicle_counts.sort_values('total', ascending=False).head(10)
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Create the stacked bar
    bottom = np.zeros(len(top_locations))
    
    for column, color in zip(['total_cars', 'total_trucks', 'total_buses', 'total_bikes', 'total_peds'],
                           ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
        plt.bar(top_locations['location_name'], top_locations[column], bottom=bottom, 
                label=column.replace('total_', '').capitalize(), color=color)
        bottom += top_locations[column]
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Count')
    plt.title('Vehicle Type Distribution at Top 10 Busiest Locations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vehicle_distribution_by_location.png', dpi=300)
    plt.close()

def create_hourly_traffic_by_vehicle():
    """Create line chart showing hourly traffic patterns by vehicle type"""
    # Group by hour and calculate mean for each vehicle type
    hourly_data = df.groupby('hour').agg({
        'total_cars': 'mean',
        'total_trucks': 'mean',
        'total_buses': 'mean',
        'total_bikes': 'mean',
        'total_peds': 'mean'
    }).reset_index()
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot each vehicle type
    plt.plot(hourly_data['hour'], hourly_data['total_cars'], marker='o', linewidth=2, label='Cars')
    plt.plot(hourly_data['hour'], hourly_data['total_trucks'], marker='s', linewidth=2, label='Trucks')
    plt.plot(hourly_data['hour'], hourly_data['total_buses'], marker='^', linewidth=2, label='Buses')
    plt.plot(hourly_data['hour'], hourly_data['total_bikes'], marker='d', linewidth=2, label='Bikes')
    plt.plot(hourly_data['hour'], hourly_data['total_peds'], marker='*', linewidth=2, label='Pedestrians')
    
    plt.title('Hourly Traffic Pattern by Vehicle Type', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Average Count', fontsize=14)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hourly_traffic_by_vehicle.png', dpi=300)
    plt.close()

def create_directional_traffic_radar():
    """Create a radar chart showing traffic flow in different directions"""
    # Calculate mean traffic for each direction
    directional_traffic = {
        'North': df[[col for col in df.columns if col.startswith('total_n_')]].sum(axis=1).mean(),
        'South': df[[col for col in df.columns if col.startswith('total_s_')]].sum(axis=1).mean(),
        'East': df[[col for col in df.columns if col.startswith('total_e_')]].sum(axis=1).mean(), 
        'West': df[[col for col in df.columns if col.startswith('total_w_')]].sum(axis=1).mean()
    }
    
    # Print the values to debug
    print("Directional traffic values:", directional_traffic)
    
    # Set up the radar chart
    categories = list(directional_traffic.keys())
    values = list(directional_traffic.values())
    
    # Close the loop
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    # Create plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angle for each direction (fixed to match the number of categories exactly)
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # The length should be the same as categories
    print(f"Length of categories: {len(categories)}, Length of angles: {len(angles)}")
    
    # Make sure we don't add an extra angle
    if len(angles) < len(categories):
        angles += angles[:1]  # Close the loop only if needed
        
    print(f"After adjustment - Length of categories: {len(categories)}, Length of angles: {len(angles)}, Length of values: {len(values)}")
    
    # Make sure both arrays are the same length
    if len(angles) > len(values):
        angles = angles[:len(values)]
    elif len(values) > len(angles):
        values = values[:len(angles)]
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set category labels (only use the original categories, not the repeated one)
    plt.xticks(angles, categories, fontsize=14)
    
    # Add labels and title
    plt.title('Directional Traffic Flow', fontsize=16)
    plt.tight_layout()
    plt.savefig('directional_traffic_radar.png', dpi=300)
    plt.close()

def create_traffic_composition_sunburst():
    """Create an interactive sunburst chart showing traffic composition"""
    # Calculate sums for each category
    total_cars = df['total_cars'].sum()
    total_trucks = df['total_trucks'].sum()
    total_buses = df['total_buses'].sum()
    total_bikes = df['total_bikes'].sum()
    total_peds = df['total_peds'].sum()
    
    # Calculate directional breakdowns
    car_directions = {
        'North': df['total_n_cars'].sum() / total_cars if total_cars > 0 else 0,
        'South': df['total_s_cars'].sum() / total_cars if total_cars > 0 else 0,
        'East': df['total_e_cars'].sum() / total_cars if total_cars > 0 else 0,
        'West': df['total_w_cars'].sum() / total_cars if total_cars > 0 else 0
    }
    
    truck_directions = {
        'North': df['total_n_trucks'].sum() / total_trucks if total_trucks > 0 else 0,
        'South': df['total_s_trucks'].sum() / total_trucks if total_trucks > 0 else 0,
        'East': df['total_e_trucks'].sum() / total_trucks if total_trucks > 0 else 0,
        'West': df['total_w_trucks'].sum() / total_trucks if total_trucks > 0 else 0
    }
    
    bus_directions = {
        'North': df['total_n_buses'].sum() / total_buses if total_buses > 0 else 0,
        'South': df['total_s_buses'].sum() / total_buses if total_buses > 0 else 0,
        'East': df['total_e_buses'].sum() / total_buses if total_buses > 0 else 0,
        'West': df['total_w_buses'].sum() / total_buses if total_buses > 0 else 0
    }
    
    # Prepare data for sunburst
    labels = ['Total Traffic', 
              'Cars', 'Trucks', 'Buses', 'Bikes', 'Pedestrians',
              'Cars-North', 'Cars-South', 'Cars-East', 'Cars-West',
              'Trucks-North', 'Trucks-South', 'Trucks-East', 'Trucks-West',
              'Buses-North', 'Buses-South', 'Buses-East', 'Buses-West']
    
    parents = ['', 
               'Total Traffic', 'Total Traffic', 'Total Traffic', 'Total Traffic', 'Total Traffic',
               'Cars', 'Cars', 'Cars', 'Cars',
               'Trucks', 'Trucks', 'Trucks', 'Trucks',
               'Buses', 'Buses', 'Buses', 'Buses']
    
    values = [total_cars + total_trucks + total_buses + total_bikes + total_peds,
              total_cars, total_trucks, total_buses, total_bikes, total_peds,
              car_directions['North'] * total_cars, car_directions['South'] * total_cars,
              car_directions['East'] * total_cars, car_directions['West'] * total_cars,
              truck_directions['North'] * total_trucks, truck_directions['South'] * total_trucks,
              truck_directions['East'] * total_trucks, truck_directions['West'] * total_trucks,
              bus_directions['North'] * total_buses, bus_directions['South'] * total_buses,
              bus_directions['East'] * total_buses, bus_directions['West'] * total_buses]
    
    # Create sunburst chart with Plotly
    fig = px.sunburst(
        names=labels,
        parents=parents,
        values=values,
        title="Traffic Composition by Vehicle Type and Direction",
        branchvalues="total"
    )
    
    fig.write_html("traffic_composition_sunburst.html")

def create_monthly_vehicle_heatmap():
    """Create a heatmap showing traffic patterns by month and vehicle type"""
    # Group by month and calculate mean for each vehicle type
    monthly_data = df.groupby('month').agg({
        'total_cars': 'mean',
        'total_trucks': 'mean',
        'total_buses': 'mean',
        'total_bikes': 'mean',
        'total_peds': 'mean'
    })
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_data.T, cmap='YlGnBu', annot=True, fmt='.1f')
    
    # Add labels and title
    plt.title('Monthly Traffic Patterns by Vehicle Type', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Vehicle Type', fontsize=14)
    plt.xticks(np.arange(0.5, 12.5), calendar.month_abbr[1:], rotation=0)
    plt.tight_layout()
    plt.savefig('monthly_vehicle_heatmap.png', dpi=300)
    plt.close()

def create_vehicle_pie_map():
    """Create a map with pie charts showing vehicle distribution at top 10 busiest locations"""
    # Get total counts by vehicle type and location
    location_vehicle_counts = df.groupby(['location_name', 'latitude', 'longitude']).agg({
        'total_cars': 'mean',
        'total_trucks': 'mean',
        'total_buses': 'mean',
        'total_bikes': 'mean',
        'total_peds': 'mean'
    }).reset_index()
    
    # Add total column
    location_vehicle_counts['total'] = location_vehicle_counts[['total_cars', 'total_trucks', 
                                                              'total_buses', 'total_bikes', 
                                                              'total_peds']].sum(axis=1)
    
    # Get top 10 locations
    top_locations = location_vehicle_counts.sort_values('total', ascending=False).head(10)
    
    # Create map
    m = folium.Map(location=[43.7001, -79.4163], zoom_start=11)
    
    # Create a directory for pie chart images if it doesn't exist
    if not os.path.exists('pie_charts'):
        os.makedirs('pie_charts')
    
    # Create pie charts for each location and add to map
    for i, row in top_locations.iterrows():
        # Extract data for pie chart
        values = [row['total_cars'], row['total_trucks'], row['total_buses'], row['total_bikes'], row['total_peds']]
        labels = ['Cars', 'Trucks', 'Buses', 'Bikes', 'Peds']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Calculate percentages for popup
        total = row['total']
        cars_pct = row['total_cars'] / total * 100
        trucks_pct = row['total_trucks'] / total * 100
        buses_pct = row['total_buses'] / total * 100
        bikes_pct = row['total_bikes'] / total * 100
        peds_pct = row['total_peds'] / total * 100
        
        # Create pie chart
        plt.figure(figsize=(4, 4))
        wedges, texts, autotexts = plt.pie(values, labels=None, autopct='%1.1f%%', 
                                           colors=colors, shadow=False)
        # Reduce font size of percentage labels
        for autotext in autotexts:
            autotext.set_fontsize(8)
        
        # Add ranking number
        plt.text(-0.2, 0.1, f"#{i+1}", fontsize=12, fontweight='bold')
        
        # Add total traffic value
        plt.text(0, -2.2, f"Total: {int(total)}", fontsize=10, ha='center')
        
        # Save pie chart
        chart_file = f'pie_charts/location_{i+1}.png'
        plt.savefig(chart_file, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Create a custom icon using the saved pie chart
        icon = folium.CustomIcon(
            icon_image=chart_file,
            icon_size=(100, 100),
            icon_anchor=(50, 50)
        )
        
        # Add marker with custom icon
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=icon,
            popup=folium.Popup(f"<strong>{row['location_name']}</strong><br>Total Traffic: {row['total']:.0f}<br>Cars: {row['total_cars']:.0f} ({cars_pct:.1f}%)<br>Trucks: {row['total_trucks']:.0f} ({trucks_pct:.1f}%)<br>Buses: {row['total_buses']:.0f} ({buses_pct:.1f}%)<br>Bikes: {row['total_bikes']:.0f} ({bikes_pct:.1f}%)<br>Pedestrians: {row['total_peds']:.0f} ({peds_pct:.1f}%)", max_width=300)
        ).add_to(m)
        
        # Add location name as a label
        short_name = row['location_name']
        if len(short_name) > 30:
            short_name = short_name[:27] + '...'
        
        folium.map.Marker(
            [row['latitude'] - 0.003, row['longitude']],
            icon=folium.DivIcon(
                icon_size=(150, 20),
                icon_anchor=(75, 0),
                html=f'<div style="font-size: 10pt; color: black; text-align: center; background-color: white; border-radius: 5px; padding: 2px;">{short_name}</div>'
            )
        ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 160px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white; padding: 10px;
                ">
      <p><strong>Vehicle Types</strong></p>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #1f77b4; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Cars</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #ff7f0e; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Trucks</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #2ca02c; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Buses</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #d62728; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Bikes</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: #9467bd; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Pedestrians</span>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save('vehicle_pie_map.html')

if __name__ == "__main__":
    print("Generating visualizations...")
    
    # Create all visualizations
    create_heatmap_by_time()
    create_vehicle_type_heatmaps()
    create_location_heatmap()
    create_top_locations_map()
    create_vehicle_distribution_by_location()
    create_hourly_traffic_by_vehicle()
    create_directional_traffic_radar()
    create_traffic_composition_sunburst()
    create_monthly_vehicle_heatmap()
    create_vehicle_pie_map()
    
    print("All visualizations have been created!") 