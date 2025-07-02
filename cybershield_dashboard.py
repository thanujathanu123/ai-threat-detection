# CyberShield Dashboard - Advanced AI-Powered Network Threat Detection
# Author: Thanu | Final Year B.Tech (Cybersecurity)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
import random
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(
    page_title="🛡️ CyberShield - AI Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .threat-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .safe-status {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .live-indicator {
        color: #ff4757;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛡️ CyberShield: Advanced Network Threat Detection</h1>', unsafe_allow_html=True)
st.markdown("### 🚀 Real-time Monitoring | 🧠 Machine Learning | 🌍 Geographic Mapping | 📧 Smart Alerts")

# Sidebar
st.sidebar.title("⚙️ Advanced Settings")
monitoring_mode = st.sidebar.selectbox("🔍 Monitoring Mode", 
                                     ["File Upload", "Real-time Simulation", "Historical Analysis", "Generate Sample Data"])
threshold = st.sidebar.slider("🎯 Alert Sensitivity (%)", min_value=50, max_value=100, value=75)
alert_email = st.sidebar.text_input("📧 Alert Email", placeholder="your@email.com")
enable_alerts = st.sidebar.checkbox("🚨 Enable Email Alerts", value=False)

# Real-time settings
if monitoring_mode == "Real-time Simulation":
    st.sidebar.subheader("🔴 Real-time Settings")
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("⏱️ Refresh Interval (seconds)", 1, 10, 3)
    freeze_map = st.sidebar.checkbox("🗺️ Freeze Map (for zooming)", value=False)

# AI Model Selection
st.sidebar.subheader("🧠 AI Model Selection")
model_type = st.sidebar.selectbox("Choose AI Model", 
                                ["Random Forest (Recommended)", "Gradient Boosting", "Auto-Select Best"])

# Advanced preprocessing function
def advanced_preprocess(df):
    """Enhanced preprocessing for network traffic data"""
    try:
        # Required columns for the model
        required_columns = ['Source Port', 'Destination Port', 'Protocol', 'Length', 
                          'TTL', 'Window Size', 'TCP Flags']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None
        
        # Select and clean the data
        X = df[required_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Convert to numeric if needed
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        return X
        
    except Exception as e:
        st.error(f"❌ Preprocessing error: {str(e)}")
        return None

# Load AI model
@st.cache_resource
def load_model():
    """Load the trained AI model"""
    try:
        if model_type == "Gradient Boosting":
            model = joblib.load('app/gradient_boost_model.pkl')
        elif model_type == "Auto-Select Best":
            # Try to load the best performing model
            try:
                model = joblib.load('app/model.pkl')  # This is the best model selected during training
            except:
                model = joblib.load('app/random_forest_model.pkl')
        else:  # Random Forest (default)
            model = joblib.load('app/random_forest_model.pkl')
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading AI model: {str(e)}")
        st.info("💡 Please run 'python create_simple_model.py' first to create the AI models.")
        return None

# Generate live data
def generate_live_data():
    """Generate realistic live network traffic data"""
    n_samples = random.randint(20, 50)
    
    data = {
        'Source Port': np.random.randint(1024, 65535, n_samples),
        'Destination Port': np.random.choice([80, 443, 22, 21, 25, 53, 8080, 3389, 1433, 5432], n_samples),
        'Protocol': np.random.choice([6, 17, 1], n_samples, p=[0.7, 0.25, 0.05]),
        'Length': np.random.lognormal(7, 1, n_samples).astype(int),
        'TTL': np.random.choice([32, 64, 128, 255], n_samples, p=[0.1, 0.6, 0.25, 0.05]),
        'Window Size': np.random.choice([1024, 2048, 4096, 8192, 16384, 32768, 65535], n_samples),
        'TCP Flags': np.random.choice([2, 16, 18, 24, 25], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'Latitude': np.random.uniform(-90, 90, n_samples),
        'Longitude': np.random.uniform(-180, 180, n_samples),
        'Country': np.random.choice(['USA', 'China', 'Russia', 'Germany', 'UK', 'India', 'Brazil'], n_samples),
        'Source IP': [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(n_samples)]
    }
    
    return pd.DataFrame(data)

# Email alert function
def send_alert_email(threats, total, email):
    """Simulate sending email alerts"""
    if email and threats > 0:
        st.sidebar.success(f"📧 Alert sent to {email}: {threats}/{total} threats detected!")

# Create geographic map
def create_geographic_map(df):
    """Create an interactive geographic map showing threat locations"""
    try:
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return None
        
        # Create base map
        m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
        
        # Add threat markers
        for idx, row in df.iterrows():
            if 'Prediction' in df.columns and row['Prediction'] == 1:
                # Threat location (red)
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=8,
                    popup=f"🚨 THREAT DETECTED<br>IP: {row.get('Source IP', 'Unknown')}<br>Country: {row.get('Country', 'Unknown')}<br>Confidence: {row.get('Confidence (%)', 'N/A')}%",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
            else:
                # Normal traffic (green)
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=4,
                    popup=f"✅ Normal Traffic<br>IP: {row.get('Source IP', 'Unknown')}<br>Country: {row.get('Country', 'Unknown')}",
                    color='green',
                    fill=True,
                    fillColor='green',
                    fillOpacity=0.5
                ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"❌ Error creating map: {str(e)}")
        return None

# Create advanced charts
def create_advanced_charts(df):
    """Create comprehensive analysis charts"""
    charts = {}
    
    # Threat confidence distribution
    if 'Confidence (%)' in df.columns and 'Prediction' in df.columns:
        fig1 = px.histogram(df, x='Confidence (%)', color='Prediction',
                           title="🎯 Threat Detection Confidence Distribution",
                           color_discrete_map={0: 'green', 1: 'red'},
                           nbins=20)
        charts['threat_confidence'] = fig1
    
    # Protocol analysis
    if 'Protocol' in df.columns:
        protocol_names = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
        df_temp = df.copy()
        df_temp['Protocol_Name'] = df_temp['Protocol'].map(protocol_names)
        
        if 'Prediction' in df.columns:
            fig2 = px.sunburst(df_temp, path=['Protocol_Name', 'Prediction'], 
                              title="📡 Protocol vs Threat Analysis")
        else:
            fig2 = px.pie(df_temp, names='Protocol_Name', title="📡 Protocol Distribution")
        charts['protocol_analysis'] = fig2
    
    # Port analysis
    if 'Destination Port' in df.columns:
        port_counts = df['Destination Port'].value_counts().head(10)
        fig3 = px.bar(x=port_counts.index, y=port_counts.values,
                     title="🔌 Top 10 Destination Ports",
                     labels={'x': 'Port', 'y': 'Count'})
        charts['port_analysis'] = fig3
    
    # Packet size analysis
    if 'Length' in df.columns and 'Prediction' in df.columns:
        fig4 = px.box(df, x='Prediction', y='Length',
                     title="📦 Packet Size Distribution by Threat Status",
                     color='Prediction',
                     color_discrete_map={0: 'green', 1: 'red'})
        charts['packet_analysis'] = fig4
    
    return charts

# Main content based on monitoring mode
if monitoring_mode == "Real-time Simulation":
    st.subheader("🔴 LIVE MONITORING")
    st.markdown('<span class="live-indicator">● LIVE</span>', unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        start_monitoring = st.button("🚀 Start Monitoring")
    with col2:
        stop_monitoring = st.button("⏹️ Stop Monitoring")
    with col3:
        single_scan = st.button("🔍 Single Scan")
    
    # Initialize session state for real-time data
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'live_data_history' not in st.session_state:
        st.session_state.live_data_history = []
    
    # Control monitoring state
    if start_monitoring:
        st.session_state.monitoring_active = True
    if stop_monitoring:
        st.session_state.monitoring_active = False
    
    # Create containers for different sections
    metrics_container = st.container()
    charts_container = st.container()
    map_container = st.container()
    
    # Generate and process data
    if single_scan or st.session_state.monitoring_active:
        # Generate live data
        live_df = generate_live_data()
        
        # Process data
        X = advanced_preprocess(live_df)
        model = load_model()
        
        if model and X is not None:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] * 100
            
            # Create results
            live_df['Prediction'] = predictions
            live_df['Confidence (%)'] = probabilities
            
            # Store in history (keep last 5 scans)
            st.session_state.live_data_history.append(live_df)
            if len(st.session_state.live_data_history) > 5:
                st.session_state.live_data_history.pop(0)
            
            # Calculate metrics
            total_connections = len(live_df)
            suspicious_connections = sum(predictions)
            detection_rate = (suspicious_connections / total_connections) * 100
            
            # Send alerts if threats detected
            if suspicious_connections > 0 and enable_alerts:
                send_alert_email(suspicious_connections, total_connections, alert_email)
            
            # Display metrics in fixed container
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔍 Total Scanned", total_connections, delta=f"+{random.randint(1,5)}")
                with col2:
                    st.metric("🚨 Threats Found", suspicious_connections, delta=f"+{random.randint(0,2)}")
                with col3:
                    st.metric("📊 Detection Rate", f"{detection_rate:.1f}%", delta=f"{random.uniform(-2,2):.1f}%")
                with col4:
                    st.metric("⚡ Processing Speed", f"{random.randint(50,100)} pkt/s", delta=f"+{random.randint(1,10)}")
            
            # Display charts in fixed container
            with charts_container:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Real-time threat detection
                    fig = px.scatter(live_df, x=range(len(live_df)), y='Confidence (%)',
                                   color='Prediction', size='Length',
                                   title="🔴 Live Threat Detection",
                                   color_discrete_map={0: 'green', 1: 'red'})
                    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{len(st.session_state.live_data_history)}")
                
                with col2:
                    # Protocol distribution
                    protocol_names = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
                    live_df['Protocol_Name'] = live_df['Protocol'].map(protocol_names)
                    fig2 = px.pie(live_df, names='Protocol_Name', title="📡 Live Protocol Mix")
                    st.plotly_chart(fig2, use_container_width=True, key=f"pie_{len(st.session_state.live_data_history)}")
            
            # Geographic map in separate container (only update if not frozen)
            with map_container:
                if 'Latitude' in live_df.columns:
                    st.subheader("🌍 Global Threat Map")
                    if not freeze_map:
                        threat_map = create_geographic_map(live_df)
                        if threat_map:
                            st_folium(threat_map, width=700, height=400, key=f"map_{len(st.session_state.live_data_history)}")
                    else:
                        st.info("🗺️ Map is frozen for interaction. Uncheck 'Freeze Map' to resume updates.")
                        if len(st.session_state.live_data_history) > 0:
                            threat_map = create_geographic_map(st.session_state.live_data_history[-1])
                            if threat_map:
                                st_folium(threat_map, width=700, height=400, key="frozen_map")
    
    # Auto-refresh logic
    if st.session_state.monitoring_active and auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

elif monitoring_mode == "Historical Analysis":
    st.subheader("📈 Historical Trend Analysis")
    
    # Generate historical data
    days = st.slider("📅 Analysis Period (days)", 1, 30, 7)
    
    if st.button("📊 Generate Historical Report"):
        # Simulate historical data
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        historical_data = []
        
        for date in dates:
            threats = random.randint(0, 50)
            total = random.randint(100, 500)
            historical_data.append({
                'Date': date,
                'Threats_Detected': threats,
                'Total_Connections': total,
                'Detection_Rate': (threats/total)*100,
                'False_Positives': random.randint(0, 5)
            })
        
        hist_df = pd.DataFrame(historical_data)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Threats", hist_df['Threats_Detected'].sum())
        with col2:
            st.metric("Avg Detection Rate", f"{hist_df['Detection_Rate'].mean():.1f}%")
        with col3:
            st.metric("Peak Threats/Hour", hist_df['Threats_Detected'].max())
        with col4:
            st.metric("False Positive Rate", f"{hist_df['False_Positives'].mean():.1f}%")

elif monitoring_mode == "File Upload":
    st.subheader("📁 Upload Network Traffic Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} records loaded.")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Records", len(df))
            with col2:
                st.metric("📋 Columns", len(df.columns))
            with col3:
                st.metric("💾 File Size", f"{uploaded_file.size} bytes")
            
            # Show data preview
            with st.expander("👀 Data Preview"):
                st.dataframe(df.head())
            
            # Column mapping
            st.subheader("🔗 Column Mapping")
            st.info("Map your CSV columns to the required network traffic features:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                source_port_col = st.selectbox("Source Port", df.columns, index=0)
                dest_port_col = st.selectbox("Destination Port", df.columns, index=1 if len(df.columns) > 1 else 0)
                protocol_col = st.selectbox("Protocol", df.columns, index=2 if len(df.columns) > 2 else 0)
                length_col = st.selectbox("Packet Length", df.columns, index=3 if len(df.columns) > 3 else 0)
            
            with col2:
                ttl_col = st.selectbox("TTL", df.columns, index=4 if len(df.columns) > 4 else 0)
                window_col = st.selectbox("Window Size", df.columns, index=5 if len(df.columns) > 5 else 0)
                flags_col = st.selectbox("TCP Flags", df.columns, index=6 if len(df.columns) > 6 else 0)
            
            if st.button("🔍 Analyze Uploaded Data", type="primary"):
                with st.spinner("🧠 AI is analyzing your network traffic..."):
                    # Create mapped dataframe
                    mapped_df = pd.DataFrame({
                        'Source Port': df[source_port_col],
                        'Destination Port': df[dest_port_col],
                        'Protocol': df[protocol_col],
                        'Length': df[length_col],
                        'TTL': df[ttl_col],
                        'Window Size': df[window_col],
                        'TCP Flags': df[flags_col]
                    })
                    
                    # Add geographic data if available
                    if 'Latitude' in df.columns and 'Longitude' in df.columns:
                        mapped_df['Latitude'] = df['Latitude']
                        mapped_df['Longitude'] = df['Longitude']
                    else:
                        # Generate random geographic data for visualization
                        mapped_df['Latitude'] = np.random.uniform(-90, 90, len(mapped_df))
                        mapped_df['Longitude'] = np.random.uniform(-180, 180, len(mapped_df))
                    
                    if 'Country' in df.columns:
                        mapped_df['Country'] = df['Country']
                    else:
                        mapped_df['Country'] = np.random.choice(['USA', 'China', 'Russia', 'Germany', 'UK'], len(mapped_df))
                    
                    if 'Source IP' in df.columns:
                        mapped_df['Source IP'] = df['Source IP']
                    else:
                        mapped_df['Source IP'] = [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(len(mapped_df))]
                    
                    # Preprocess and predict
                    X = advanced_preprocess(mapped_df)
                    model = load_model()
                    
                    if model and X is not None:
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)[:, 1] * 100
                        
                        # Add results to dataframe
                        mapped_df['Prediction'] = predictions
                        mapped_df['Confidence (%)'] = probabilities
                        
                        # Calculate metrics
                        total_connections = len(mapped_df)
                        suspicious_connections = sum(predictions)
                        detection_rate = (suspicious_connections / total_connections) * 100
                        
                        # Display results
                        st.success("🎉 Analysis Complete!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🔍 Total Analyzed", total_connections)
                        with col2:
                            st.metric("🚨 Threats Detected", suspicious_connections)
                        with col3:
                            st.metric("📊 Detection Rate", f"{detection_rate:.1f}%")
                        with col4:
                            avg_confidence = mapped_df[mapped_df['Prediction'] == 1]['Confidence (%)'].mean()
                            st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%" if not pd.isna(avg_confidence) else "N/A")
                        
                        # Visualizations
                        charts = create_advanced_charts(mapped_df)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'threat_confidence' in charts:
                                st.plotly_chart(charts['threat_confidence'], use_container_width=True)
                        with col2:
                            if 'protocol_analysis' in charts:
                                st.plotly_chart(charts['protocol_analysis'], use_container_width=True)
                        
                        # Geographic map
                        if 'Latitude' in mapped_df.columns:
                            st.subheader("🌍 Geographic Threat Distribution")
                            threat_map = create_geographic_map(mapped_df)
                            if threat_map:
                                st_folium(threat_map, width=700, height=400)
                        
                        # Detailed results
                        with st.expander("📋 Detailed Analysis Results"):
                            st.dataframe(mapped_df)
                        
                        # Download results
                        if suspicious_connections > 0:
                            suspicious = mapped_df[mapped_df['Prediction'] == 1]
                            csv = suspicious.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Threat Report",
                                data=csv,
                                file_name=f"threat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("✅ No suspicious connections to display.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

elif monitoring_mode == "Generate Sample Data":
    st.subheader("🎲 Generate Custom Sample Data")
    st.info("Create custom network traffic datasets for testing and demonstration purposes.")
    
    # Sample data configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Configuration")
        sample_size = st.slider("📈 Number of Records", 50, 2000, 500)
        threat_percentage = st.slider("🚨 Threat Percentage", 5, 80, 25)
        include_geo = st.checkbox("🌍 Include Geographic Data", value=True)
        include_timestamps = st.checkbox("⏰ Include Timestamps", value=True)
    
    with col2:
        st.subheader("🎯 Attack Types to Include")
        port_scanning = st.checkbox("🔍 Port Scanning Attacks", value=True)
        ddos_attacks = st.checkbox("💥 DDoS Patterns", value=True)
        malware_traffic = st.checkbox("🦠 Malware Communication", value=True)
        data_exfiltration = st.checkbox("📤 Data Exfiltration", value=True)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            protocol_mix = st.selectbox("📡 Protocol Distribution", 
                                      ["Balanced (TCP/UDP/ICMP)", "TCP Heavy", "UDP Heavy", "Mixed"])
            packet_size_dist = st.selectbox("📦 Packet Size Distribution", 
                                          ["Normal", "Large Packets", "Small Packets", "Mixed"])
        with col2:
            time_pattern = st.selectbox("⏰ Time Pattern", 
                                      ["Random", "Business Hours", "Night Activity", "Weekend Pattern"])
            geographic_focus = st.selectbox("🌍 Geographic Focus", 
                                          ["Global", "North America", "Europe", "Asia", "Suspicious Regions"])
    
    # Generate button
    if st.button("🎲 Generate Sample Dataset", type="primary"):
        with st.spinner("🔄 Generating custom network traffic data..."):
            # Generate custom data based on parameters
            np.random.seed(42)  # For reproducible results
            
            # Base data structure
            data = {
                'Source Port': np.random.randint(1024, 65535, sample_size),
                'Destination Port': [],
                'Protocol': [],
                'Length': [],
                'TTL': np.random.choice([32, 64, 128, 255], sample_size, p=[0.1, 0.6, 0.25, 0.05]),
                'Window Size': np.random.choice([1024, 2048, 4096, 8192, 16384, 32768, 65535], sample_size),
                'TCP Flags': []
            }
            
            # Configure based on selections
            if protocol_mix == "TCP Heavy":
                protocols = np.random.choice([6, 17, 1], sample_size, p=[0.8, 0.15, 0.05])
            elif protocol_mix == "UDP Heavy":
                protocols = np.random.choice([6, 17, 1], sample_size, p=[0.3, 0.65, 0.05])
            else:  # Balanced or Mixed
                protocols = np.random.choice([6, 17, 1], sample_size, p=[0.6, 0.3, 0.1])
            
            data['Protocol'] = protocols
            
            # Destination ports based on attack types
            common_ports = [80, 443, 22, 53]
            attack_ports = []
            if port_scanning:
                attack_ports.extend([21, 23, 25, 110, 143, 993, 995])
            if ddos_attacks:
                attack_ports.extend([80, 443, 53])
            if malware_traffic:
                attack_ports.extend([4444, 6666, 8080, 9999])
            if data_exfiltration:
                attack_ports.extend([21, 22, 443, 993])
            
            all_ports = common_ports + attack_ports
            data['Destination Port'] = np.random.choice(all_ports, sample_size)
            
            # Packet sizes
            if packet_size_dist == "Large Packets":
                data['Length'] = np.random.lognormal(8, 0.5, sample_size).astype(int)
            elif packet_size_dist == "Small Packets":
                data['Length'] = np.random.lognormal(6, 0.5, sample_size).astype(int)
            else:  # Normal or Mixed
                data['Length'] = np.random.lognormal(7, 1, sample_size).astype(int)
            
            # TCP Flags
            data['TCP Flags'] = np.random.choice([2, 16, 18, 24, 25], sample_size, p=[0.1, 0.3, 0.4, 0.15, 0.05])
            
            # Add timestamps if requested
            if include_timestamps:
                if time_pattern == "Business Hours":
                    # Generate timestamps during business hours
                    base_time = datetime.now().replace(hour=9, minute=0, second=0)
                    timestamps = [base_time + timedelta(seconds=random.randint(0, 8*3600)) for _ in range(sample_size)]
                elif time_pattern == "Night Activity":
                    base_time = datetime.now().replace(hour=22, minute=0, second=0)
                    timestamps = [base_time + timedelta(seconds=random.randint(0, 6*3600)) for _ in range(sample_size)]
                else:  # Random or Weekend
                    timestamps = [datetime.now() - timedelta(seconds=random.randint(0, 86400*7)) for _ in range(sample_size)]
                
                data['Timestamp'] = timestamps
            
            # Add geographic data if requested
            if include_geo:
                if geographic_focus == "North America":
                    data['Latitude'] = np.random.uniform(25, 60, sample_size)
                    data['Longitude'] = np.random.uniform(-130, -60, sample_size)
                    data['Country'] = np.random.choice(['USA', 'Canada', 'Mexico'], sample_size)
                elif geographic_focus == "Europe":
                    data['Latitude'] = np.random.uniform(35, 70, sample_size)
                    data['Longitude'] = np.random.uniform(-10, 40, sample_size)
                    data['Country'] = np.random.choice(['Germany', 'UK', 'France', 'Italy', 'Spain'], sample_size)
                elif geographic_focus == "Asia":
                    data['Latitude'] = np.random.uniform(10, 50, sample_size)
                    data['Longitude'] = np.random.uniform(70, 140, sample_size)
                    data['Country'] = np.random.choice(['China', 'India', 'Japan', 'South Korea'], sample_size)
                elif geographic_focus == "Suspicious Regions":
                    data['Latitude'] = np.random.uniform(-90, 90, sample_size)
                    data['Longitude'] = np.random.uniform(-180, 180, sample_size)
                    data['Country'] = np.random.choice(['Unknown', 'Tor Exit', 'VPN', 'Proxy'], sample_size)
                else:  # Global
                    data['Latitude'] = np.random.uniform(-90, 90, sample_size)
                    data['Longitude'] = np.random.uniform(-180, 180, sample_size)
                    data['Country'] = np.random.choice(['USA', 'China', 'Russia', 'Germany', 'UK', 'India', 'Brazil'], sample_size)
                
                data['Source IP'] = [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(sample_size)]
            
            # Create DataFrame
            df_generated = pd.DataFrame(data)
            
            # Generate labels based on threat percentage
            target_threats = int(sample_size * threat_percentage / 100)
            labels = [1] * target_threats + [0] * (sample_size - target_threats)
            random.shuffle(labels)
            df_generated['Label'] = labels
            
            # Process with AI model
            X = advanced_preprocess(df_generated)
            model = load_model()
            
            if model and X is not None:
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1] * 100
                
                df_generated['AI_Prediction'] = predictions
                df_generated['Confidence (%)'] = probabilities
        
        # Display results
        st.success(f"✅ Generated {sample_size} network traffic records!")
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", sample_size)
        with col2:
            actual_threats = df_generated['Label'].sum()
            st.metric("🚨 Labeled Threats", actual_threats)
        with col3:
            if model and X is not None:
                ai_threats = df_generated['AI_Prediction'].sum()
                st.metric("🤖 AI Detected", ai_threats)
            else:
                st.metric("🤖 AI Detected", "N/A")
        with col4:
            if model and X is not None:
                accuracy = accuracy_score(df_generated['Label'], df_generated['AI_Prediction'])
                st.metric("🎯 AI Accuracy", f"{accuracy:.1%}")
            else:
                st.metric("🎯 AI Accuracy", "N/A")
        
        # Show data preview
        with st.expander("👀 Data Preview"):
            st.dataframe(df_generated.head(20))
        
        # Download options
        st.subheader("💾 Download Generated Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = df_generated.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv_data,
                file_name=f"generated_network_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Only threats
            threats_only = df_generated[df_generated['Label'] == 1]
            if len(threats_only) > 0:
                threats_csv = threats_only.to_csv(index=False)
                st.download_button(
                    label="🚨 Download Threats Only (CSV)",
                    data=threats_csv,
                    file_name=f"threats_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Analysis report
            report = f"""
# Network Traffic Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- Total Records: {sample_size}
- Threat Percentage: {threat_percentage}%
- Actual Threats: {df_generated['Label'].sum()}
- Geographic Data: {'Yes' if include_geo else 'No'}
- Timestamps: {'Yes' if include_timestamps else 'No'}

## Configuration
- Protocol Mix: {protocol_mix}
- Packet Size: {packet_size_dist}
- Time Pattern: {time_pattern}
- Geographic Focus: {geographic_focus}

## Attack Types Included
- Port Scanning: {'Yes' if port_scanning else 'No'}
- DDoS Patterns: {'Yes' if ddos_attacks else 'No'}
- Malware Traffic: {'Yes' if malware_traffic else 'No'}
- Data Exfiltration: {'Yes' if data_exfiltration else 'No'}
"""
            
            st.download_button(
                label="📋 Download Analysis Report",
                data=report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Visualization of generated data
        if len(df_generated) > 0:
            st.subheader("📊 Generated Data Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Threat distribution
                threat_dist = df_generated['Label'].value_counts()
                fig1 = px.pie(values=threat_dist.values, names=['Normal', 'Threat'], 
                             title="🛡️ Generated Threat Distribution",
                             color_discrete_map={0: 'green', 1: 'red'})
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Protocol distribution
                protocol_names = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
                df_generated['Protocol_Name'] = df_generated['Protocol'].map(protocol_names)
                fig2 = px.histogram(df_generated, x='Protocol_Name', color='Label',
                                  title="📡 Protocol Distribution by Threat Status",
                                  color_discrete_map={0: 'green', 1: 'red'})
                st.plotly_chart(fig2, use_container_width=True)
            
            # Geographic visualization if available
            if include_geo and 'Latitude' in df_generated.columns:
                st.subheader("🌍 Geographic Distribution")
                threat_map = create_geographic_map(df_generated)
                if threat_map:
                    st_folium(threat_map, width=700, height=400)

# Footer
st.markdown("---")
st.markdown("### 🛡️ CyberShield | Built with ❤️ by Thanu | B.Tech (Cybersecurity)")