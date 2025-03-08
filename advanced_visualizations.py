import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create a folder for advanced visualizations
output_folder = 'advanced_visualizations'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the data
print("Loading data...")
df = pd.read_csv('data/University_Dataset(tmc_raw_data_2020_2029).csv')

# Convert datetime columns
df['count_date'] = pd.to_datetime(df['count_date'])
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['day_of_week'] = df['count_date'].dt.dayofweek
df['day_name'] = df['count_date'].dt.day_name()
df['month'] = df['count_date'].dt.month
df['month_name'] = df['count_date'].dt.month_name()
df['hour'] = df['start_time'].dt.hour
df['year'] = df['count_date'].dt.year

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

print("Data preprocessing complete.")

def create_3d_traffic_surface():
    """Create a 3D surface plot of traffic by hour and day of week"""
    print("Creating 3D traffic surface plot...")
    
    # Create pivot table
    pivot_data = df.pivot_table(
        values='total_traffic', 
        index='hour',
        columns='day_name',
        aggfunc='mean'
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    available_days = [day for day in day_order if day in pivot_data.columns]
    pivot_data = pivot_data[available_days]
    
    # Create 3D surface plot with Plotly
    fig = go.Figure(data=[go.Surface(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Viridis',
        colorbar=dict(title='Average Traffic Volume')
    )])
    
    fig.update_layout(
        title='3D Traffic Volume by Hour and Day of Week',
        scene=dict(
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            zaxis_title='Traffic Volume',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1)
            )
        ),
        width=1000,
        height=800
    )
    
    fig.write_html(f"{output_folder}/3d_traffic_surface.html")
    print("3D traffic surface plot created.")

def create_clustering_map():
    """Create a map showing location clusters based on traffic patterns"""
    print("Creating traffic pattern clusters map...")
    
    # Aggregate data by location and hour
    hourly_patterns = df.groupby(['location_name', 'hour'])['total_traffic'].mean().unstack().reset_index()
    
    # Check how many complete rows we have before filtering
    before_count = len(hourly_patterns)
    
    # Only include locations with complete hourly data
    hourly_patterns = hourly_patterns.dropna()
    
    # Check how many rows remain after filtering
    after_count = len(hourly_patterns)
    print(f"Found {before_count} locations, {after_count} with complete hourly data")
    
    # If we don't have enough samples for clustering, try a different approach
    if after_count < 5:
        print("Not enough complete data for clustering. Using a more flexible approach.")
        
        # Calculate total traffic by location - make sure we're calculating this directly from df
        location_traffic = df.groupby('location_name')['total_traffic'].mean().reset_index()
        
        # Get coordinates for each location
        location_coords = df.groupby('location_name').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        # Merge traffic data with coordinates
        location_data = pd.merge(location_traffic, location_coords, on='location_name')
        
        # Filter out any locations with NaN values
        location_data = location_data.dropna()
        
        # Make sure we have at least one location
        if len(location_data) == 0:
            print("No valid location data found. Skipping clustering map.")
            return
            
        # Create traffic categories
        num_categories = min(5, len(location_data))
        try:
            location_data['cluster'] = pd.qcut(
                location_data['total_traffic'], 
                q=num_categories, 
                labels=False, 
                duplicates='drop'
            )
        except ValueError:
            # If qcut fails (e.g., all values identical), use simple categories
            print("Could not create quantile-based categories. Using simple categories.")
            location_data['cluster'] = 0
            num_categories = 1
        
        # Create map
        m = folium.Map(location=[43.7001, -79.4163], zoom_start=11)
        
        # Define colors for clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        # Add a legend
        legend_html = '<div style="position: fixed; bottom: 50px; right: 50px; width: 240px; z-index:9999; font-size:14px; background-color:white; padding:10px; border:2px solid grey;">'
        legend_html += '<p><strong>Traffic Volume Categories</strong></p>'
        
        # Add each cluster to the map and legend
        for i in range(num_categories):
            cluster_data = location_data[location_data['cluster'] == i]
            color = colors[i % len(colors)]
            
            # Add to legend
            legend_html += f'<div style="display:flex; align-items:center;"><div style="background-color:{color}; width:20px; height:20px; margin-right:5px;"></div>'
            legend_html += f'<span>Category {i+1}: {len(cluster_data)} locations</span></div>'
            
            # Add markers to map
            for _, row in cluster_data.iterrows():
                folium.CircleMarker(
                    location=(row['latitude'], row['longitude']),
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Location: {row['location_name']}<br>Traffic: {row['total_traffic']:.0f}"
                ).add_to(m)
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        m.save(f'{output_folder}/traffic_categories_map.html')
        
        print("Traffic categories map created.")
        return
    
    # If we have enough samples, proceed with clustering as before
    # Keep location names as a reference
    locations = hourly_patterns['location_name']
    
    # Prepare data for clustering (exclude location name)
    X = hourly_patterns.drop('location_name', axis=1)
    
    # Determine appropriate number of clusters
    n_clusters = min(5, len(X))
    print(f"Using {n_clusters} clusters for {len(X)} locations")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster information to the dataframe
    hourly_patterns['cluster'] = clusters
    
    # Get coordinates for each location
    location_coords = df.groupby('location_name').agg({
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    # Merge with cluster information
    clustered_locations = pd.merge(
        hourly_patterns[['location_name', 'cluster']], 
        location_coords,
        on='location_name'
    )
    
    # Create map
    m = folium.Map(location=[43.7001, -79.4163], zoom_start=11)
    
    # Define colors for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'black', 'white']
    
    # Add a legend
    legend_html = '<div style="position: fixed; bottom: 50px; right: 50px; width: 200px; z-index:9999; font-size:14px; background-color:white; padding:10px; border:2px solid grey;">'
    legend_html += '<p><strong>Traffic Pattern Clusters</strong></p>'
    
    # Add each cluster to the map and legend
    for i in range(n_clusters):
        cluster_data = clustered_locations[clustered_locations['cluster'] == i]
        color = colors[i % len(colors)]
        
        # Add to legend
        legend_html += f'<div style="display:flex; align-items:center;"><div style="background-color:{color}; width:20px; height:20px; margin-right:5px;"></div>'
        legend_html += f'<span>Cluster {i+1}: {len(cluster_data)} locations</span></div>'
        
        # Add markers to map
        for _, row in cluster_data.iterrows():
            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Location: {row['location_name']}<br>Cluster: {i+1}"
            ).add_to(m)
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(f'{output_folder}/traffic_clusters_map.html')
    
    # Only create pattern plots if we have sufficient clusters
    if n_clusters > 1:
        # Plot the cluster centers (average traffic patterns)
        cluster_centers = []
        for i in range(n_clusters):
            cluster_data = hourly_patterns[hourly_patterns['cluster'] == i].drop(['location_name', 'cluster'], axis=1)
            cluster_centers.append(cluster_data.mean())
        
        # Arrange in a dataframe for plotting
        cluster_centers_df = pd.DataFrame(cluster_centers)
        
        # Create a subplot figure
        fig, axes = plt.subplots(n_clusters, 1, figsize=(14, 4*n_clusters))
        
        # Ensure axes is always a list/array for consistency
        if n_clusters == 1:
            axes = [axes]
        
        # Plot each cluster's pattern
        for i in range(n_clusters):
            ax = axes[i]
            ax.plot(cluster_centers_df.iloc[i], 'o-', linewidth=2, color=colors[i])
            ax.set_title(f'Cluster {i+1}: {sum(clusters == i)} locations', fontsize=14)
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Average Traffic', fontsize=12)
            ax.set_xticks(range(0, 24))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/traffic_clusters_patterns.png', dpi=300)
        plt.close()
    
    print("Traffic pattern clusters map created.")

def create_directional_flow_diagram():
    """Create a Sankey diagram showing directional traffic flows instead of Chord"""
    print("Creating directional traffic flow diagram...")
    
    # Calculate total traffic in each direction
    directional_totals = {
        'North': df[[col for col in df.columns if col.startswith('total_n_')]].sum().sum(),
        'South': df[[col for col in df.columns if col.startswith('total_s_')]].sum().sum(),
        'East': df[[col for col in df.columns if col.startswith('total_e_')]].sum().sum(),
        'West': df[[col for col in df.columns if col.startswith('total_w_')]].sum().sum()
    }
    
    # Create relationship matrix (approximate flows between directions)
    matrix = np.array([
        [0, directional_totals['North'] * 0.4, directional_totals['North'] * 0.3, directional_totals['North'] * 0.3],
        [directional_totals['South'] * 0.4, 0, directional_totals['South'] * 0.3, directional_totals['South'] * 0.3],
        [directional_totals['East'] * 0.3, directional_totals['East'] * 0.3, 0, directional_totals['East'] * 0.4],
        [directional_totals['West'] * 0.3, directional_totals['West'] * 0.3, directional_totals['West'] * 0.4, 0]
    ])
    
    # Create Sankey diagram data
    sources = []
    targets = []
    values = []
    labels = ["North", "South", "East", "West"]
    
    # Create the flows
    for i in range(4):
        for j in range(4):
            if i != j and matrix[i, j] > 0:
                sources.append(i)
                targets.append(j)
                values.append(matrix[i, j] / 1000)  # Scale down for better visualization
    
    # Create Sankey diagram using Plotly
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["red", "blue", "green", "orange"]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(
        title="Traffic Flow Between Directions (in thousands)",
        font=dict(size=14),
        width=800,
        height=600
    )
    
    # Also create a simple bar chart of directional traffic for backup
    plt.figure(figsize=(10, 6))
    plt.bar(directional_totals.keys(), directional_totals.values(), color=['red', 'blue', 'green', 'orange'])
    plt.title('Total Traffic by Direction')
    plt.ylabel('Total Traffic Volume')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{output_folder}/directional_traffic_bar.png', dpi=300)
    plt.close()
    
    # Try to save the Sankey diagram, with error handling
    try:
        fig.write_html(f"{output_folder}/directional_flow_diagram.html")
        print("Directional flow diagram created.")
    except Exception as e:
        print(f"Error creating Sankey diagram: {e}")
        print("Created bar chart as an alternative.")

def create_turning_movement_analysis():
    """Create visualizations that analyze turning movements at busy intersections"""
    print("Creating turning movement analysis...")
    
    # Get average turning movements for each location
    turning_data = df.groupby('location_name').agg({
        'n_appr_cars_r': 'mean',  # Right turn from North
        'n_appr_cars_t': 'mean',  # Through from North
        'n_appr_cars_l': 'mean',  # Left turn from North
        's_appr_cars_r': 'mean',  # Right turn from South
        's_appr_cars_t': 'mean',  # Through from South
        's_appr_cars_l': 'mean',  # Left turn from South
        'e_appr_cars_r': 'mean',  # Right turn from East
        'e_appr_cars_t': 'mean',  # Through from East
        'e_appr_cars_l': 'mean',  # Left turn from East
        'w_appr_cars_r': 'mean',  # Right turn from West
        'w_appr_cars_t': 'mean',  # Through from West
        'w_appr_cars_l': 'mean',  # Left turn from West
        'latitude': 'first',
        'longitude': 'first',
        'total_traffic': 'mean'
    }).reset_index()
    
    # Get top 9 busiest locations for a 3x3 grid
    top_locations = turning_data.sort_values('total_traffic', ascending=False).head(9)
    
    # Create a compact visualization of turning movements for top locations
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Color map for arrows
    colors = ['green', 'blue', 'red']
    
    for i, (_, loc) in enumerate(top_locations.iterrows()):
        ax = axes[i]
        
        # Create a basic intersection diagram
        ax.plot([-1, 1], [0, 0], 'k-', linewidth=2)  # East-West road
        ax.plot([0, 0], [-1, 1], 'k-', linewidth=2)  # North-South road
        
        # Direction labels
        ax.text(0, 1.1, 'N', ha='center', fontsize=10)
        ax.text(0, -1.1, 'S', ha='center', fontsize=10)
        ax.text(1.1, 0, 'E', va='center', fontsize=10)
        ax.text(-1.1, 0, 'W', va='center', fontsize=10)
        
        # Define movements with start points, end points, and values
        movements = [
            # North approach
            ((0, 0.5), (-0.5, 0), loc['n_appr_cars_l']),  # Left
            ((0, 0.5), (0, -0.5), loc['n_appr_cars_t']),  # Through
            ((0, 0.5), (0.5, 0), loc['n_appr_cars_r']),   # Right
            
            # South approach
            ((0, -0.5), (0.5, 0), loc['s_appr_cars_l']),  # Left
            ((0, -0.5), (0, 0.5), loc['s_appr_cars_t']),  # Through
            ((0, -0.5), (-0.5, 0), loc['s_appr_cars_r']), # Right
            
            # East approach
            ((0.5, 0), (0, 0.5), loc['e_appr_cars_l']),   # Left
            ((0.5, 0), (-0.5, 0), loc['e_appr_cars_t']),  # Through
            ((0.5, 0), (0, -0.5), loc['e_appr_cars_r']),  # Right
            
            # West approach
            ((-0.5, 0), (0, -0.5), loc['w_appr_cars_l']), # Left
            ((-0.5, 0), (0.5, 0), loc['w_appr_cars_t']),  # Through
            ((-0.5, 0), (0, 0.5), loc['w_appr_cars_r'])   # Right
        ]
        
        # Draw movements
        for j, (start, end, value) in enumerate(movements):
            # Skip if no data
            if pd.isna(value) or value == 0:
                continue
                
            # Scale width by value
            width = 0.5 + 4.5 * (value / top_locations[['n_appr_cars_t', 's_appr_cars_t', 'e_appr_cars_t', 'w_appr_cars_t']].max().max())
            
            # Draw curved arrow
            color_idx = j % 3  # Different color for left, through, right
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=width, 
                                      connectionstyle='arc3,rad=0.3', 
                                      color=colors[color_idx], alpha=0.7))
        
        # Set title and limits
        short_name = loc['location_name']
        if len(short_name) > 30:
            short_name = short_name[:27] + '...'
        ax.set_title(f"{i+1}. {short_name}", fontsize=10)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Add a legend for the turning movements
    legend_ax = fig.add_axes([0.92, 0.5, 0.05, 0.4])
    legend_ax.axis('off')
    legend_ax.plot([0, 0.2], [0.8, 0.8], '-', color=colors[0], linewidth=2, label='Left Turn')
    legend_ax.plot([0, 0.2], [0.6, 0.6], '-', color=colors[1], linewidth=2, label='Through')
    legend_ax.plot([0, 0.2], [0.4, 0.4], '-', color=colors[2], linewidth=2, label='Right Turn')
    legend_ax.legend(fontsize=12)
    
    plt.suptitle('Turning Movement Patterns at Top 9 Busiest Intersections', fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f'{output_folder}/turning_movements.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Turning movement analysis created.")

def create_pedestrian_safety_map():
    """Create a map highlighting pedestrian safety concerns"""
    print("Creating pedestrian safety map...")
    
    # Calculate pedestrian to vehicle ratios
    df['ped_vehicle_ratio'] = df['total_peds'] / (df['total_cars'] + df['total_trucks'] + df['total_buses'] + 0.1)
    
    # Calculate average pedestrian counts and vehicle traffic by location
    ped_safety = df.groupby('location_name').agg({
        'total_peds': 'mean',
        'total_cars': 'mean',
        'total_trucks': 'mean',
        'total_buses': 'mean',
        'ped_vehicle_ratio': 'mean',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    # Calculate total vehicles
    ped_safety['total_vehicles'] = ped_safety['total_cars'] + ped_safety['total_trucks'] + ped_safety['total_buses']
    
    # Filter for locations with significant pedestrian activity
    ped_safety = ped_safety[ped_safety['total_peds'] > 10]
    
    # Create a scatter plot of pedestrian vs vehicle volume
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        ped_safety['total_vehicles'], 
        ped_safety['total_peds'],
        c=ped_safety['ped_vehicle_ratio'], 
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Pedestrian/Vehicle Ratio')
    
    # Add labels for some notable points
    for i, row in ped_safety.nlargest(10, 'ped_vehicle_ratio').iterrows():
        plt.annotate(
            row['location_name'].split('/')[0],  # Just use first part of location name
            xy=(row['total_vehicles'], row['total_peds']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Average Vehicle Volume (log scale)')
    plt.ylabel('Average Pedestrian Volume (log scale)')
    plt.title('Pedestrian vs Vehicle Traffic Volume')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/pedestrian_vs_vehicle.png', dpi=300)
    plt.close()
    
    # Create a map showing pedestrian safety concerns
    m = folium.Map(location=[43.7001, -79.4163], zoom_start=11)
    
    # Define thresholds for categorization
    vehicle_high = ped_safety['total_vehicles'].quantile(0.75)
    ped_high = ped_safety['total_peds'].quantile(0.75)
    ped_medium = ped_safety['total_peds'].quantile(0.5)
    
    # Add locations to map
    for _, row in ped_safety.iterrows():
        # Create popup content
        popup_content = f"""
        <strong>{row['location_name']}</strong><br>
        Pedestrians: {row['total_peds']:.1f}<br>
        Vehicles: {row['total_vehicles']:.1f}<br>
        Ped/Vehicle Ratio: {row['ped_vehicle_ratio']:.3f}
        """
        
        # Determine risk level and color
        if row['total_vehicles'] > vehicle_high and row['total_peds'] > ped_high:
            # High risk: high vehicle and pedestrian volume
            color = 'red'
        elif row['total_vehicles'] > vehicle_high and row['total_peds'] > ped_medium:
            # Medium risk: high vehicle and medium pedestrian volume
            color = 'orange'
        elif row['total_peds'] > ped_high and row['ped_vehicle_ratio'] > 0.5:
            # Low risk but high pedestrian activity
            color = 'green'
        else:
            color = 'blue'
            
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_content
        ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 300px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white; padding: 10px;
                ">
      <p><strong>Pedestrian Safety Concerns</strong></p>
      <div style="display: flex; align-items: center;">
        <div style="background-color: red; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>High Risk: High Vehicle + High Pedestrian Volume</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: orange; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Medium Risk: High Vehicle + Medium Pedestrian</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: green; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Pedestrian Priority Area: High Pedestrian Activity</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="background-color: blue; width: 20px; height: 20px; margin-right: 5px;"></div>
        <span>Other Locations</span>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(f'{output_folder}/pedestrian_safety_map.html')
    print("Pedestrian safety map created.")

if __name__ == "__main__":
    print(f"Generating advanced visualizations in '{output_folder}' folder...")
    
    # Create all visualizations
    create_3d_traffic_surface()
    create_clustering_map()
    create_directional_flow_diagram()
    create_turning_movement_analysis()
    create_pedestrian_safety_map()
    
    print("All advanced visualizations have been created!") 