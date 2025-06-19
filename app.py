import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import os
from datetime import datetime, timedelta
from cyber_geo_analyzer import CybersecurityLogAnalyzer





# Set page configuration
st.set_page_config(
    page_title="Cybersecurity Log Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .anomaly-row {
        background-color: #ff6b6b;
        color: white;
    }
    .normal-row {
        background-color: #51cf66;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .geoip-section {
        background-color: #222b36;
        color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed cybersecurity logs"""
    try:
        # Try multiple possible file names
        possible_files = [
            'processed_cybersecurity_logs.csv',
            'cybersecurity_system_logs.csv',
            'logs.csv'
        ]
        
        df = None
        for filename in possible_files:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                st.success(f"✅ Loaded data from {filename}")
                break
        
        if df is None:
            st.error("❌ No data file found! Please ensure your CSV file is in the same directory.")
            st.info("Expected files: processed_cybersecurity_logs.csv, cybersecurity_system_logs.csv, or logs.csv")
            return None
        
        # Handle timestamp column
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            except:
                st.warning("Could not parse Timestamp column, using current time")
                df['Timestamp'] = pd.Timestamp.now()
        else:
            st.warning("No Timestamp column found, creating dummy timestamps")
            df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        
        # Check if processed columns exist, if not create them
        if 'Predicted_Anomaly' not in df.columns:
            st.warning("No ML predictions found. Run cybersecurity_log_analyzer.py first for full functionality.")
            # Create dummy predictions for demo purposes
            np.random.seed(42)
            df['Predicted_Anomaly'] = np.random.choice([-1, 1], size=len(df), p=[0.15, 0.85])
            df['Anomaly_Score'] = np.random.uniform(-0.5, 0.5, size=len(df))
        
        # Check for GeoIP columns
        if 'GeoCity' not in df.columns or 'GeoCountry' not in df.columns:
            st.info("🌍 GeoIP data not found. Run the analyzer with GeoIP enrichment for enhanced geographical analysis.")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def parse_natural_language_query(query, df):
    """
    Parse natural language queries and filter the dataframe
    
    Args:
        query (str): Natural language query
        df (DataFrame): Dataset to filter
    
    Returns:
        DataFrame: Filtered dataset
    """
    if not query:
        return df
    
    query = query.lower()
    filtered_df = df.copy()
    
    # Parse anomalies
    if 'anomal' in query:
        filtered_df = filtered_df[filtered_df['Predicted_Anomaly'] == -1]
    
    # Parse specific users
    user_match = re.search(r'user[_\s]*(\d+|[a-zA-Z0-9_]+)', query)
    if user_match:
        user_id = user_match.group(1)
        # Try to find exact match or partial match
        user_filter = filtered_df['User_ID'].str.contains(user_id, case=False, na=False)
        filtered_df = filtered_df[user_filter]
    
    
    # Parse locations/cities
    location_keywords = ['mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad', 'pune', 'kolkata']
    for location in location_keywords:
        if location in query:
            # Initialize with matching index
            location_filter = pd.Series(False, index=filtered_df.index)

            if 'Location' in filtered_df.columns:
                location_filter |= filtered_df['Location'].str.contains(location, case=False, na=False)

            if 'GeoCity' in filtered_df.columns:
                location_filter |= filtered_df['GeoCity'].str.contains(location, case=False, na=False)

            # Apply filter safely
            location_filter = location_filter.reindex(filtered_df.index, fill_value=False)
            filtered_df = filtered_df[location_filter]
            break

    
    # Parse countries for GeoIP
    country_keywords = ['usa', 'china', 'russia', 'india', 'uk', 'germany', 'france', 'japan']
    for country in country_keywords:
        if country in query:
            if 'GeoCountry' in filtered_df.columns:
                country_filter = filtered_df['GeoCountry'].str.contains(country, case=False, na=False)
                filtered_df = filtered_df[country_filter]
            break
    
    # Parse session duration
    session_match = re.search(r'session[>\s]*(\d+)', query)
    if session_match:
        duration = int(session_match.group(1))
        if '>' in query:
            filtered_df = filtered_df[filtered_df['Session_Duration'] > duration]
        else:
            filtered_df = filtered_df[filtered_df['Session_Duration'] == duration]
    
    # Parse attack types
    attack_types = ['brute_force', 'sql_injection', 'xss', 'ddos', 'malware']
    for attack in attack_types:
        if attack.replace('_', ' ') in query or attack in query:
            attack_filter = filtered_df['Attack_Type'].str.contains(attack, case=False, na=False)
            filtered_df = filtered_df[attack_filter]
    
    return filtered_df

def create_kpi_metrics(df):
    total_logs = len(df)
    anomalies = len(df[df['Predicted_Anomaly'] == -1])
    anomaly_rate = (anomalies / total_logs * 100) if total_logs > 0 else 0
    unique_users = df['User_ID'].nunique()
    unique_locations = df['Location'].nunique() if 'Location' in df.columns else 0
    avg_session = df['Session_Duration'].mean() if 'Session_Duration' in df.columns else 0
    
    # GeoIP metrics
    unique_countries = df['GeoCountry'].nunique() if 'GeoCountry' in df.columns else 0
    unique_cities = df['GeoCity'].nunique() if 'GeoCity' in df.columns else 0

    st.markdown("""
        <style>
        .kpi-light {
            background-color: #f9f9f9;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .kpi-light h2 {
            font-size: 1.8em;
            margin: 0;
            color: #343a40;
        }
        .kpi-light p {
            margin: 5px 0 0;
            color: #666666;
            font-size: 0.9em;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        st.markdown(f"""<div class="kpi-light"><h2>📊 {total_logs:,}</h2><p>Total Logs</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-light"><h2>🚨 {anomalies:,}</h2><p>Anomalies ({anomaly_rate:.1f}%)</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-light"><h2>👥 {unique_users}</h2><p>Unique Users</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-light"><h2>🌍 {unique_locations}</h2><p>Locations</p></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="kpi-light"><h2>⏱️ {avg_session:.1f}m</h2><p>Avg Session</p></div>""", unsafe_allow_html=True)
    with col6:
        if unique_countries > 0:
            st.markdown(f"""<div class="kpi-light"><h2>🏳️ {unique_countries}</h2><p>Countries</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="kpi-light"><h2>🏳️ N/A</h2><p>Countries</p></div>""", unsafe_allow_html=True)
    with col7:
        if unique_cities > 0:
            st.markdown(f"""<div class="kpi-light"><h2>🏙️ {unique_cities}</h2><p>Cities</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="kpi-light"><h2>🏙️ N/A</h2><p>Cities</p></div>""", unsafe_allow_html=True)

def create_geoip_world_map(df):
    """Create a world map showing anomalies by country"""
    if 'GeoCountry' not in df.columns:
        return None
    
    # Count anomalies by country
    anomaly_by_country = df[df['Predicted_Anomaly'] == -1]['GeoCountry'].value_counts().reset_index()
    anomaly_by_country.columns = ['Country', 'Anomalies']
    
    # Create world map
    fig = px.choropleth(
        anomaly_by_country,
        locations='Country',
        color='Anomalies',
        hover_name='Country',
        color_continuous_scale='Reds',
        title="Global Threat Distribution",
        locationmode='ISO-3'
    )
    
    fig.update_layout(height=500)
    return fig

def create_geoip_city_chart(df):
    """Create bar chart for top cities by anomalies"""
    if 'GeoCity' not in df.columns:
        return None
    
    # Get top 15 cities by anomaly count
    anomaly_by_city = df[df['Predicted_Anomaly'] == -1]['GeoCity'].value_counts().head(15)
    
    fig = px.bar(
        x=anomaly_by_city.values,
        y=anomaly_by_city.index,
        orientation='h',
        title="Top 15 Cities by Anomaly Count",
        labels={'x': 'Number of Anomalies', 'y': 'City'},
        color=anomaly_by_city.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    return fig

def create_geoip_anomaly_rate_chart(df):
    """Create chart showing anomaly rates by country"""
    if 'GeoCountry' not in df.columns:
        return None
    
    # Calculate anomaly rates by country (only for countries with >10 logs)
    country_stats = df.groupby('GeoCountry').agg({
        'Predicted_Anomaly': ['count', lambda x: (x == -1).sum()]
    }).round(3)
    
    country_stats.columns = ['Total_Logs', 'Anomalies']
    country_stats['Anomaly_Rate'] = (country_stats['Anomalies'] / country_stats['Total_Logs'] * 100).round(2)
    
    # Filter countries with at least 10 logs
    country_stats = country_stats[country_stats['Total_Logs'] >= 10]
    country_stats = country_stats.sort_values('Anomaly_Rate', ascending=False).head(15)
    
    fig = px.bar(
        x=country_stats.index,
        y=country_stats['Anomaly_Rate'],
        title="Top 15 Countries by Anomaly Rate (min 10 logs)",
        labels={'x': 'Country', 'y': 'Anomaly Rate (%)'},
        color=country_stats['Anomaly_Rate'],
        color_continuous_scale='OrRd'
    )
    
def create_attack_type_pie_chart(df):
    """Create pie chart for attack type distribution"""
    attack_counts = df['Attack_Type'].value_counts()
    
    fig = px.pie(
        values=attack_counts.values,
        names=attack_counts.index,
        title="Attack Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_anomaly_by_location_chart(df):
    """Create bar chart for anomalies by location"""
    anomaly_df = df[df['Predicted_Anomaly'] == -1]
    
    # Use GeoCity if available, otherwise fall back to Location
    if 'GeoCity' in df.columns and df['GeoCity'].notna().any():
        location_counts = anomaly_df['GeoCity'].value_counts().head(10)
        title = "Top 10 Cities by Anomaly Count"
        x_label = "City"
    else:
        location_counts = anomaly_df['Location'].value_counts().head(10)
        title = "Top 10 Locations by Anomaly Count"
        x_label = "Location"

    # Convert the Series to a DataFrame for proper coloring
    location_counts_df = location_counts.reset_index()
    location_counts_df.columns = [x_label, 'Anomaly_Count']

    fig = px.bar(
        location_counts_df,
        x=x_label,
        y='Anomaly_Count',
        title=title,
        labels={x_label: x_label, 'Anomaly_Count': 'Number of Anomalies'},
        color='Anomaly_Count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)

    return fig

def create_timeline_chart(df):
    """Create timeline chart showing anomalies over time"""
    df_timeline = df.copy()
    df_timeline['Date'] = df_timeline['Timestamp'].dt.date
    
    timeline_data = df_timeline.groupby(['Date', 'Predicted_Anomaly']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    if 1 in timeline_data.columns:
        fig.add_trace(go.Scatter(
            x=timeline_data.index,
            y=timeline_data[1],
            mode='lines+markers',
            name='Normal Logs',
            line=dict(color='green', width=2)
        ))
    
    if -1 in timeline_data.columns:
        fig.add_trace(go.Scatter(
            x=timeline_data.index,
            y=timeline_data[-1],
            mode='lines+markers',
            name='Anomalous Logs',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title="Log Activity Timeline",
        xaxis_title="Date",
        yaxis_title="Number of Logs",
        height=400
    )
    
    return fig

def run_geoip_enrichment():
    """Run GeoIP enrichment on the current dataset"""
    with st.spinner("🌍 Enriching data with GeoIP information..."):
        try:
            analyzer = CybersecurityLogAnalyzer()
            if analyzer.load_data() and analyzer.preprocess_data():
                success = analyzer.enrich_with_geoip()  # Removed batch_size and max_workers
                if success:
                    analyzer.save_model_and_data()
                    st.success("✅ GeoIP enrichment completed! Please refresh the page to see updated data.")
                    st.balloons()
                else:
                    st.error("❌ GeoIP enrichment failed. Check your internet connection and try again.")
            else:
                st.error("❌ Could not load or preprocess data for GeoIP enrichment.")
        except Exception as e:
            st.error(f"❌ Error during GeoIP enrichment: {str(e)}")

st.write("🧪 Debug: has GeoIP method?", hasattr(CybersecurityLogAnalyzer(), 'enrich_with_geoip'))

def main():
    st.title("🔐 Cybersecurity Log Analyzer Dashboard")
    st.markdown("*Enhanced with GeoIP Intelligence*")

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Check if GeoIP enrichment is available
    has_geoip = 'GeoCity' in df.columns and 'GeoCountry' in df.columns
    
    # GeoIP Control Panel
    if not has_geoip:
        st.markdown("""
        <div class="geoip-section">
            <h3>🌍 GeoIP Enrichment Available</h3>
            <p>Enhance your analysis with geographical intelligence! Click below to enrich your data with city and country information for each IP address.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start GeoIP Enrichment", type="primary", use_container_width=True):
                run_geoip_enrichment()
                st.stop()


    st.markdown("---")
    st.subheader("🧠 Run Anomaly Detection (ML Model)")

    if st.button("🧠 Run Anomaly Detection"):
        analyzer = CybersecurityLogAnalyzer(csv_file_path='processed_cybersecurity_logs.csv')
        if analyzer.load_data() and analyzer.preprocess_data() and analyzer.train_isolation_forest():
            analyzer.save_model_and_data(data_path='processed_cybersecurity_logs.csv')
            st.success("Anomaly detection completed and data saved!")
        else:
            st.error("❌ Failed to run anomaly detection. Please check your log data.")

            run_geoip_enrichment()
            st.stop()
    else:
        st.success("✅ Dataset includes GeoIP enrichment")

    # Display KPI metrics
    create_kpi_metrics(df)

    # Sidebar filters
    st.sidebar.header("🔧 Filters & Controls")
    
    # User filter
    user_options = ['All Users'] + sorted(df['User_ID'].unique().tolist())
    selected_user = st.sidebar.selectbox("Select User:", user_options)
    
    # Location filter - use GeoCity if available
    if has_geoip:
        location_options = ['All Locations'] + sorted(df['GeoCity'].dropna().unique().tolist())
        selected_location = st.sidebar.selectbox("Select City:", location_options)
        
        # Country filter
        country_options = ['All Countries'] + sorted(df['GeoCountry'].dropna().unique().tolist())
        selected_country = st.sidebar.selectbox("Select Country:", country_options)
    else:
        location_options = ['All Locations'] + sorted(df['Location'].unique().tolist())
        selected_location = st.sidebar.selectbox("Select Location:", location_options)
        selected_country = 'All Countries'
    
    # Anomaly filter
    anomaly_filter = st.sidebar.selectbox(
        "Show:",
        ["All Logs", "Only Anomalies", "Only Normal Logs"]
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(df['Timestamp'].min().date(), df['Timestamp'].max().date()),
        min_value=df['Timestamp'].min().date(),
        max_value=df['Timestamp'].max().date()
    )
    
    # Natural Language Query
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗣️ Natural Language Query")
    nl_query = st.sidebar.text_input(
        "Enter query:",
        placeholder="e.g., 'Show anomalies from USA' or 'List user_42 logs with session > 60'"
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply sidebar filters
    if selected_user != 'All Users':
        filtered_df = filtered_df[filtered_df['User_ID'] == selected_user]
    
    if has_geoip:
        if selected_location != 'All Locations':
            filtered_df = filtered_df[filtered_df['GeoCity'] == selected_location]
        if selected_country != 'All Countries':
            filtered_df = filtered_df[filtered_df['GeoCountry'] == selected_country]
    else:
        if selected_location != 'All Locations':
            filtered_df = filtered_df[filtered_df['Location'] == selected_location]
    
    if anomaly_filter == "Only Anomalies":
        filtered_df = filtered_df[filtered_df['Predicted_Anomaly'] == -1]
    elif anomaly_filter == "Only Normal Logs":
        filtered_df = filtered_df[filtered_df['Predicted_Anomaly'] == 1]
    
    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Timestamp'].dt.date >= start_date) & 
            (filtered_df['Timestamp'].dt.date <= end_date)
        ]
    
    # Apply natural language query
    if nl_query:
        filtered_df = parse_natural_language_query(nl_query, filtered_df)
        st.info(f"🔍 Query: '{nl_query}' - Found {len(filtered_df)} matching logs")
    
    st.markdown("---")
    
    # Charts section
    if has_geoip:
        # GeoIP-enhanced visualizations
        st.subheader("🌍 Global Threat Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            world_map = create_geoip_world_map(filtered_df)
            if world_map:
                st.plotly_chart(world_map, use_container_width=True)
            else:
                st.info("World map requires country data")
        
        with col2:
            city_chart = create_geoip_city_chart(filtered_df)
            if city_chart:
                st.plotly_chart(city_chart, use_container_width=True)
            else:
                st.info("City chart requires GeoIP data")
        
        # Anomaly rates by country
        anomaly_rate_chart = create_geoip_anomaly_rate_chart(filtered_df)
        if anomaly_rate_chart:
            st.plotly_chart(anomaly_rate_chart, use_container_width=True)
    
    # Standard charts
    st.subheader("📊 Attack Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_attack_type_pie_chart(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_anomaly_by_location_chart(filtered_df), use_container_width=True)
    
    # Timeline chart
    st.plotly_chart(create_timeline_chart(filtered_df), use_container_width=True)
    
    # Data table section
    st.markdown("---")
    st.subheader("📋 Log Data")
    
    # Display options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        default_columns = ['Timestamp', 'User_ID', 'Location', 'Attack_Type', 'Predicted_Anomaly', 'Anomaly_Score']
        if has_geoip:
            default_columns.extend(['GeoCity', 'GeoCountry'])
        
        show_columns = st.multiselect(
            "Select columns to display:",
            options=filtered_df.columns.tolist(),
            default=[col for col in default_columns if col in filtered_df.columns]
        )
    
    with col2:
        max_rows = st.number_input("Max rows to display:", min_value=10, max_value=1000, value=100)
    
    with col3:
        if st.button("💾 Download CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data",
                data=csv,
                file_name=f"filtered_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Display logs with styling
    if show_columns:
        display_df = filtered_df[show_columns].head(max_rows)
        
        st.subheader(f"Raw Logs ({len(display_df)} of {len(filtered_df)} shown)")
        
        def highlight_anomalies(row):
            if row['Predicted_Anomaly'] == -1:
                return ['background-color: #ff6b6b; color: white;'] * len(row)
            else:
                return ['background-color: #51cf66; color: white;'] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_anomalies, axis=1),
            use_container_width=True,
            height=500
        )

    # Advanced analysis section
    st.markdown("---")
    st.subheader("📊 Advanced Threat Intelligence")

    if has_geoip:
        col_left, col_right = st.columns(2)
        
        # Top threat countries
        with col_left:
            st.write("**🏳️ Top 5 Threat Countries**")
            threat_countries = filtered_df[filtered_df['Predicted_Anomaly'] == -1]['GeoCountry'].value_counts().head(5)
            if not threat_countries.empty:
                st.bar_chart(threat_countries)
            else:
                st.write("No threat data available.")
        
        # Geographic anomaly distribution
        with col_right:
            st.write("**🌆 Geographic Risk Assessment**")
            if len(filtered_df) > 0:
                geo_risk = (
                    filtered_df.groupby('GeoCountry')['Predicted_Anomaly']
                    .apply(lambda x: (x == -1).mean())
                    .sort_values(ascending=False)
                    .head(5)
                    .round(3)
                )
                if not geo_risk.empty:
                    st.bar_chart(geo_risk)
                else:
                    st.write("No geographic risk data available.")
    else:
        col_left, col_right = st.columns(2)
        
        # Top IPs with most anomalies
        with col_left:
            st.write("**🔌 Top 5 IPs with Anomalies**")
            top_ips = filtered_df[filtered_df['Predicted_Anomaly'] == -1]['IP_Address'].value_counts().head(5)
            if not top_ips.empty:
                st.bar_chart(top_ips)
            else:
                st.write("No anomalous IPs in current filter.")

        # Locations with highest anomaly ratio
        with col_right:
            st.write("**📍 Top 5 Locations by Anomaly Rate**")
            anomaly_rate_by_location = (
                filtered_df.groupby('Location')['Predicted_Anomaly']
                .apply(lambda x: (x == -1).mean())
                .sort_values(ascending=False)
                .head(5)
                .round(2)
            )
            if not anomaly_rate_by_location.empty:
                st.bar_chart(anomaly_rate_by_location)
            else:
                st.write("No location-based anomalies found.")

    # Session analysis
    st.markdown("### ⏱️ Session Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Longest Sessions (All Logs)**")
        if 'Session_Duration' in filtered_df.columns:
            longest_sessions = (
                filtered_df.sort_values(by='Session_Duration', ascending=False)
                [['User_ID', 'Session_Duration', 'Predicted_Anomaly'] + 
                 (['GeoCity', 'GeoCountry'] if has_geoip else ['Location'])]
                .head(5)
            )
            st.dataframe(longest_sessions, use_container_width=True)
        else:
            st.write("No session duration data available.")
    
    with col2:
        st.write("**Anomalous Sessions Distribution**")
        if 'Session_Duration' in filtered_df.columns:
            anomaly_sessions = filtered_df[filtered_df['Predicted_Anomaly'] == -1]['Session_Duration']
            if not anomaly_sessions.empty:
                fig = px.histogram(
                    anomaly_sessions, 
                    nbins=20, 
                    title="Distribution of Anomalous Session Durations",
                    labels={'value': 'Session Duration (minutes)', 'count': 'Frequency'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No anomalous sessions found.")
        else:
            st.write("No session duration data available.")

    # Footer
    st.markdown("---")
    geoip_status = "with GeoIP Intelligence" if has_geoip else "| Run GeoIP enrichment for enhanced analysis"
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🔒 Cybersecurity Log Analyzer {geoip_status} | Powered by Isolation Forest ML Model
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()