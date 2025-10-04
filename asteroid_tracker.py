import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import math
import time

# Configure the page
st.set_page_config(
    page_title="Asteroid Tracker",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NASA API key - Using your provided key
NASA_API_KEY = "cuJa44ZaVIzwXPIVfcdlx74NJD3kpstdXQ4JJ7Cw"

class AsteroidTracker:
    def __init__(self):
        self.base_url = "https://api.nasa.gov/neo/rest/v1"
    
    def get_asteroids_by_date(self, start_date, end_date):
        """Fetch asteroids data for a date range with better error handling"""
        url = f"{self.base_url}/feed"
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'api_key': NASA_API_KEY
        }
        
        try:
            time.sleep(0.5)
            response = requests.get(url, params=params)
            
            if response.status_code == 429:
                st.error("🚫 API rate limit exceeded. Please wait a moment and try again.")
                return None
            elif response.status_code == 403:
                st.error("🔐 API key invalid or expired.")
                return None
            elif response.status_code != 200:
                st.error(f"API Error {response.status_code}")
                return None
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {e}")
            return self._get_demo_data()
    
    def _get_demo_data(self):
        """Provide demo data when API fails"""
        st.warning("Using demo data for display.")
        
        today = datetime.now().date()
        demo_data = {
            'near_earth_objects': {
                str(today): [
                    {
                        'id': 'demo_001',
                        'name': 'Asteroid 2025 Demo',
                        'estimated_diameter': {
                            'meters': {
                                'estimated_diameter_min': 50.0,
                                'estimated_diameter_max': 150.0
                            }
                        },
                        'is_potentially_hazardous_asteroid': False,
                        'close_approach_data': [{
                            'relative_velocity': {'kilometers_per_hour': '45000'},
                            'miss_distance': {'kilometers': '5000000'},
                            'orbiting_body': 'Earth'
                        }]
                    },
                    {
                        'id': 'demo_002', 
                        'name': 'Asteroid 2025 Hazardous',
                        'estimated_diameter': {
                            'meters': {
                                'estimated_diameter_min': 200.0,
                                'estimated_diameter_max': 500.0
                            }
                        },
                        'is_potentially_hazardous_asteroid': True,
                        'close_approach_data': [{
                            'relative_velocity': {'kilometers_per_hour': '65000'},
                            'miss_distance': {'kilometers': '1500000'},
                            'orbiting_body': 'Earth'
                        }]
                    }
                ]
            }
        }
        return demo_data
    
    def get_asteroid_details(self, asteroid_id):
        """Get detailed information about a specific asteroid"""
        if asteroid_id.startswith('demo_'):
            return None
            
        url = f"{self.base_url}/neo/{asteroid_id}"
        params = {'api_key': NASA_API_KEY}
        
        try:
            time.sleep(0.5)
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching asteroid details: {e}")
            return None

class AsteroidSimulator:
    def __init__(self):
        self.earth_radius = 6371
        self.moon_distance = 384400
    
    def create_3d_simulation(self, asteroid_data, selected_asteroid=None):
        earth = self._create_earth_sphere()
        asteroids_traces, asteroid_info = self._create_asteroid_data(asteroid_data, selected_asteroid)
        moon_trace = self._create_moon_orbit()
        
        all_traces = [earth, moon_trace] + asteroids_traces
        
        fig = go.Figure(data=all_traces)
        fig.update_layout(
            title="3D Asteroid Simulation - Earth-Centered View",
            scene=dict(
                xaxis_title="X (1000 km)",
                yaxis_title="Y (1000 km)", 
                zaxis_title="Z (1000 km)",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(eye=dict(x=2, y=2, z=2)),
                bgcolor='black'
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig, asteroid_info
    
    def _create_earth_sphere(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        
        x = self.earth_radius / 1000 * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius / 1000 * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius / 1000 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        colorscale = [[0, 'darkblue'], [0.5, 'blue'], [1, 'lightblue']]
        
        return go.Surface(
            x=x, y=y, z=z,
            colorscale=colorscale,
            showscale=False,
            opacity=0.8,
            name="Earth"
        )
    
    def _create_moon_orbit(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = (self.moon_distance / 1000) * np.cos(theta)
        y = (self.moon_distance / 1000) * np.sin(theta)
        z = np.zeros_like(theta)
        
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name="Moon Orbit",
            hoverinfo='name'
        )
    
    def _create_asteroid_data(self, asteroid_data, selected_asteroid=None):
        traces = []
        asteroid_info = {}
        
        if not asteroid_data or 'near_earth_objects' not in asteroid_data:
            return traces, asteroid_info
        
        all_asteroids = []
        for date in asteroid_data['near_earth_objects']:
            for asteroid in asteroid_data['near_earth_objects'][date]:
                all_asteroids.append(asteroid)
        
        display_asteroids = all_asteroids[:10]
        
        for i, asteroid in enumerate(display_asteroids):
            asteroid_id = asteroid['id']
            name = asteroid['name']
            is_selected = (selected_asteroid and asteroid_id == selected_asteroid)
            is_hazardous = asteroid['is_potentially_hazardous_asteroid']
            
            close_approach = asteroid['close_approach_data'][0]
            miss_distance = float(close_approach['miss_distance']['kilometers'])
            velocity = float(close_approach['relative_velocity']['kilometers_per_hour'])
            
            distance = max(miss_distance / 1000, 10)
            
            orbit_trace, position_trace, position = self._create_asteroid_orbit(
                distance, velocity, name, is_hazardous, is_selected
            )
            traces.append(orbit_trace)
            traces.append(position_trace)
            
            asteroid_info[asteroid_id] = {
                'name': name,
                'distance_km': miss_distance,
                'velocity_kmh': velocity,
                'diameter_min': asteroid['estimated_diameter']['meters']['estimated_diameter_min'],
                'diameter_max': asteroid['estimated_diameter']['meters']['estimated_diameter_max'],
                'is_hazardous': is_hazardous,
                'position': position
            }
            
            if is_selected:
                velocity_trace = self._create_velocity_vector(position, velocity, name)
                traces.append(velocity_trace)
        
        return traces, asteroid_info
    
    def _create_asteroid_orbit(self, distance, velocity, name, is_hazardous, is_selected):
        theta = np.linspace(0, 2 * np.pi, 50)
        inclination = np.radians(30 + np.random.uniform(-15, 15))
        
        x_orbit = distance * np.cos(theta)
        y_orbit = distance * np.sin(theta) * np.cos(inclination)
        z_orbit = distance * np.sin(theta) * np.sin(inclination)
        
        current_angle = np.random.uniform(0, 2 * np.pi)
        x_pos = distance * np.cos(current_angle)
        y_pos = distance * np.sin(current_angle) * np.cos(inclination)
        z_pos = distance * np.sin(current_angle) * np.sin(inclination)
        
        if is_selected:
            color = 'yellow'
            width = 4
            size = 8
        elif is_hazardous:
            color = 'red'
            width = 3
            size = 6
        else:
            color = 'orange'
            width = 2
            size = 4
        
        orbit_trace = go.Scatter3d(
            x=x_orbit, y=y_orbit, z=z_orbit,
            mode='lines',
            line=dict(color=color, width=width),
            name=f"{name} Orbit",
            hoverinfo='name',
            showlegend=False
        )
        
        position_trace = go.Scatter3d(
            x=[x_pos], y=[y_pos], z=[z_pos],
            mode='markers',
            marker=dict(size=size, color=color, symbol='circle'),
            name=name,
            hoverinfo='name',
            showlegend=False
        )
        
        return orbit_trace, position_trace, (x_pos, y_pos, z_pos)
    
    def _create_velocity_vector(self, position, velocity, name):
        x, y, z = position
        scale = min(velocity / 50000, 2.0)
        direction = np.array([-y, x, z * 0.5])
        direction = direction / np.linalg.norm(direction)
        
        x_end = x + direction[0] * scale
        y_end = y + direction[1] * scale
        z_end = z + direction[2] * scale
        
        return go.Scatter3d(
            x=[x, x_end], y=[y, y_end], z=[z, z_end],
            mode='lines+markers',
            line=dict(color='cyan', width=4),
            marker=dict(size=3, color='cyan', symbol='arrow'),
            name=f"{name} Velocity",
            hoverinfo='name',
            showlegend=False
        )

def setup_sidebar():
    st.sidebar.title("🌌 Asteroid Tracker")
    st.sidebar.markdown("---")
    
    today = datetime.now().date()
    default_start = today - timedelta(days=3)
    
    start_date = st.sidebar.date_input("Start Date", value=default_start, max_value=today)
    end_date = st.sidebar.date_input("End Date", value=today, max_value=today, min_value=start_date)
    
    date_range = (end_date - start_date).days
    if date_range > 7:
        st.sidebar.warning("⚠️ Large date ranges may hit API limits.")
    
    show_hazardous_only = st.sidebar.checkbox("Show Potentially Hazardous Only")
    min_size = st.sidebar.slider("Minimum Estimated Diameter (meters)", 0, 5000, 0, 10)
    
    return start_date, end_date, show_hazardous_only, min_size

def display_asteroid_data(data, show_hazardous_only, min_size):
    if not data or 'near_earth_objects' not in data:
        st.warning("No asteroid data available for the selected dates.")
        return
    
    asteroids = []
    for date in data['near_earth_objects']:
        for asteroid in data['near_earth_objects'][date]:
            asteroids.append({
                'id': asteroid['id'],
                'name': asteroid['name'],
                'date': date,
                'estimated_diameter_min': asteroid['estimated_diameter']['meters']['estimated_diameter_min'],
                'estimated_diameter_max': asteroid['estimated_diameter']['meters']['estimated_diameter_max'],
                'estimated_diameter_avg': (asteroid['estimated_diameter']['meters']['estimated_diameter_min'] + 
                                         asteroid['estimated_diameter']['meters']['estimated_diameter_max']) / 2,
                'is_potentially_hazardous': asteroid['is_potentially_hazardous_asteroid'],
                'relative_velocity_kmh': float(asteroid['close_approach_data'][0]['relative_velocity']['kilometers_per_hour']),
                'miss_distance_km': float(asteroid['close_approach_data'][0]['miss_distance']['kilometers']),
                'orbiting_body': asteroid['close_approach_data'][0]['orbiting_body']
            })
    
    df = pd.DataFrame(asteroids)
    
    if show_hazardous_only:
        df = df[df['is_potentially_hazardous'] == True]
    
    df = df[df['estimated_diameter_avg'] >= min_size]
    
    if df.empty:
        st.warning("No asteroids match your filter criteria.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Asteroids", len(df))
    with col2:
        hazardous_count = df['is_potentially_hazardous'].sum()
        st.metric("Potentially Hazardous", hazardous_count)
    with col3:
        avg_size = df['estimated_diameter_avg'].mean()
        st.metric("Avg Diameter (m)", f"{avg_size:.1f}")
    with col4:
        closest_approach = df['miss_distance_km'].min()
        st.metric("Closest Approach (km)", f"{closest_approach:,.0f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌍 3D Simulation", "📊 Data Table", "📈 Size Distribution", "🚀 Close Approaches", "🔬 Orbital Data"])
    
    with tab1:
        display_3d_simulation(df, data)
    with tab2:
        display_data_table(df)
    with tab3:
        display_size_charts(df)
    with tab4:
        display_approach_charts(df)
    with tab5:
        display_orbital_info(df)

def display_3d_simulation(df, raw_data):
    st.subheader("Interactive 3D Asteroid Simulation")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 🎯 Select Asteroid")
        asteroid_options = [""] + df['name'].tolist()
        selected_asteroid_name = st.selectbox("Choose an asteroid to highlight:", asteroid_options)
        
        if selected_asteroid_name:
            selected_asteroid = df[df['name'] == selected_asteroid_name].iloc[0]
            asteroid_id = selected_asteroid['id']
            
            st.markdown("### 📊 Asteroid Info")
            st.write(f"**Name:** {selected_asteroid['name']}")
            st.write(f"**Diameter:** {selected_asteroid['estimated_diameter_avg']:.1f} m")
            st.write(f"**Velocity:** {selected_asteroid['relative_velocity_kmh']:,.0f} km/h")
            st.write(f"**Miss Distance:** {selected_asteroid['miss_distance_km']:,.0f} km")
            st.write(f"**Hazardous:** {'✅ Yes' if selected_asteroid['is_potentially_hazardous'] else '❌ No'}")
        else:
            asteroid_id = None
            st.info("Select an asteroid from the list to see detailed information.")
        
        st.markdown("### 🌙 Scale Reference")
        st.write("• Earth diameter: ~12,742 km")
        st.write("• Moon distance: ~384,400 km")
    
    with col1:
        simulator = AsteroidSimulator()
        fig, asteroid_info = simulator.create_3d_simulation(raw_data, asteroid_id)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎮 Simulation Guide")
        col_guide1, col_guide2 = st.columns(2)
        with col_guide1:
            st.write("**Colors:**")
            st.write("• 🌍 Blue: Earth")
            st.write("• 🔴 Red: Hazardous")
            st.write("• 🟠 Orange: Safe")
            st.write("• 💛 Yellow: Selected")
        with col_guide2:
            st.write("**Controls:**")
            st.write("• Drag: Rotate")
            st.write("• Scroll: Zoom")
            st.write("• Shift+Drag: Pan")

def display_data_table(df):
    display_df = df[['name', 'date', 'estimated_diameter_avg', 
                    'relative_velocity_kmh', 'miss_distance_km', 
                    'is_potentially_hazardous']].copy()
    
    display_df.columns = ['Name', 'Date', 'Avg Diameter (m)', 'Velocity (km/h)', 
                         'Miss Distance (km)', 'Potentially Hazardous']
    
    display_df['Avg Diameter (m)'] = display_df['Avg Diameter (m)'].round(1)
    display_df['Velocity (km/h)'] = display_df['Velocity (km/h)'].round(0)
    display_df['Miss Distance (km)'] = display_df['Miss Distance (km)'].round(0)
    
    st.dataframe(display_df, use_container_width=True)

def display_size_charts(df):
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='estimated_diameter_avg', 
                          title='Asteroid Size Distribution',
                          labels={'estimated_diameter_avg': 'Estimated Diameter (meters)'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='is_potentially_hazardous', y='estimated_diameter_avg',
                    title='Size Comparison: Hazardous vs Non-Hazardous',
                    labels={'is_potentially_hazardous': 'Potentially Hazardous',
                           'estimated_diameter_avg': 'Estimated Diameter (meters)'})
        st.plotly_chart(fig, use_container_width=True)

def display_approach_charts(df):
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='estimated_diameter_avg', y='relative_velocity_kmh',
                        color='is_potentially_hazardous',
                        title='Velocity vs Size',
                        labels={'estimated_diameter_avg': 'Diameter (m)',
                               'relative_velocity_kmh': 'Velocity (km/h)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='miss_distance_km', 
                          title='Miss Distance Distribution',
                          labels={'miss_distance_km': 'Miss Distance (km)'})
        st.plotly_chart(fig, use_container_width=True)

def display_orbital_info(df):
    st.subheader("Detailed Asteroid Information")
    
    asteroid_names = df['name'].unique()
    selected_asteroid = st.selectbox("Select an asteroid for detailed information:", asteroid_names)
    
    if selected_asteroid:
        asteroid_data = df[df['name'] == selected_asteroid].iloc[0]
        tracker = AsteroidTracker()
        details = tracker.get_asteroid_details(asteroid_data['id'])
        
        if details:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information**")
                st.write(f"**Name:** {details['name']}")
                st.write(f"**ID:** {details['id']}")
                st.write(f"**Potentially Hazardous:** {'Yes' if details['is_potentially_hazardous_asteroid'] else 'No'}")
                
                diameter = details['estimated_diameter']['meters']
                st.write(f"**Diameter Range:** {diameter['estimated_diameter_min']:.1f} - {diameter['estimated_diameter_max']:.1f} meters")
            
            with col2:
                st.markdown("**Orbital Data**")
                orbital_data = details['orbital_data']
                st.write(f"**Orbit Class:** {orbital_data['orbit_class']['orbit_class_type']}")
                st.write(f"**Orbital Period:** {orbital_data['orbital_period']} days")
                st.write(f"**Inclination:** {orbital_data['inclination']}°")
                st.write(f"**Eccentricity:** {orbital_data['eccentricity']}")

def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌌 NASA Asteroid Tracker with 3D Simulation</h1>', unsafe_allow_html=True)
    
    st.success(f"✅ NASA API Key Active: {NASA_API_KEY[:8]}...")
    st.info("💡 **Tip**: Use smaller date ranges (3-7 days) to avoid rate limits.")
    
    tracker = AsteroidTracker()
    start_date, end_date, show_hazardous_only, min_size = setup_sidebar()
    
    with st.spinner("Loading asteroid data from NASA..."):
        data = tracker.get_asteroids_by_date(str(start_date), str(end_date))
    
    if data:
        display_asteroid_data(data, show_hazardous_only, min_size)
    
    st.markdown("---")
    st.markdown("Data provided by NASA's Near Earth Object Web Service API")
    st.markdown("🌠 Explore the cosmos safely from Earth!")

if __name__ == "__main__":
    main()
