# src/real_time_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import threading
import time
import queue
from datetime import datetime

class DashboardUpdater:
    def __init__(self, result_queue, update_interval=1.0):
        """
        Initialize the dashboard updater
        
        Args:
            result_queue: Queue with processed packet results
            update_interval: Interval in seconds to update dashboard
        """
        self.result_queue = result_queue
        self.update_interval = update_interval
        self.stop_updating = threading.Event()
        self.data_buffer = pd.DataFrame()
        self.lock = threading.Lock()
        
        # Initialize session state for real-time data if not exists
        if 'real_time_data' not in st.session_state:
            st.session_state.real_time_data = pd.DataFrame()
        if 'last_update_time' not in st.session_state:
            st.session_state.last_update_time = datetime.now()
        if 'alert_count' not in st.session_state:
            st.session_state.alert_count = 0
        if 'total_packets' not in st.session_state:
            st.session_state.total_packets = 0
            
    def update_dashboard_data(self):
        """Update dashboard data from result queue"""
        # Get all available data from queue
        new_data_frames = []
        try:
            while True:
                df = self.result_queue.get(block=False)
                new_data_frames.append(df)
                self.result_queue.task_done()
        except queue.Empty:
            pass
            
        # If no new data, return
        if not new_data_frames:
            return False
            
        # Combine all new data
        new_data = pd.concat(new_data_frames, ignore_index=True)
        
        with self.lock:
            # Update session state with new data
            if st.session_state.real_time_data.empty:
                st.session_state.real_time_data = new_data
            else:
                # Keep only the last 1000 packets to prevent memory issues
                st.session_state.real_time_data = pd.concat(
                    [st.session_state.real_time_data, new_data], 
                    ignore_index=True
                ).tail(1000)
                
            # Update statistics
            st.session_state.total_packets += len(new_data)
            st.session_state.alert_count += len(new_data[new_data['Prediction'] == 1])
            st.session_state.last_update_time = datetime.now()
            
        return True
        
    def start_updating(self):
        """Start updating dashboard in a separate thread"""
        update_thread = threading.Thread(target=self._update_loop)
        update_thread.daemon = True
        update_thread.start()
        return update_thread
        
    def _update_loop(self):
        """Update loop to periodically update dashboard data"""
        while not self.stop_updating.is_set():
            self.update_dashboard_data()
            time.sleep(self.update_interval)
            
    def stop(self):
        """Stop updating dashboard"""
        self.stop_updating.set()

def render_real_time_dashboard(threshold=85):
    """
    Render real-time dashboard components
    
    Args:
        threshold: Confidence threshold for alerts (%)
    """
    # Display last update time
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Real-time Monitoring")
    st.sidebar.write(f"Last update: {st.session_state.last_update_time.strftime('%H:%M:%S')}")
    
    # Display statistics
    st.sidebar.metric("Total Packets", st.session_state.total_packets)
    st.sidebar.metric("Alert Count", st.session_state.alert_count)
    
    # Main dashboard area
    if st.session_state.real_time_data.empty:
        st.info("Waiting for network traffic data...")
        return
        
    # Get data from session state
    df = st.session_state.real_time_data
    
    # Filter based on threshold
    suspicious = df[(df['Prediction'] == 1) & (df['Confidence (%)'] >= threshold)]
    
    # Display suspicious traffic
    st.subheader("🚨 Detected Suspicious Traffic (Real-time)")
    if not suspicious.empty:
        st.dataframe(suspicious)
        st.success(f"{len(suspicious)} suspicious connections detected!")
    else:
        st.info("No suspicious activity detected with current threshold.")
    
    # Charts
    st.subheader("📊 Real-time Traffic Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        label_count = df['Prediction'].value_counts().rename({0: 'Normal', 1: 'Attack'})
        fig = px.pie(values=label_count.values, names=label_count.index, title="Traffic Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(df, x='Confidence (%)', color='Prediction',
                          color_discrete_map={0: 'green', 1: 'red'},
                          nbins=20, title="Confidence of Threat Predictions")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Display recent traffic (last 10 packets)
    st.subheader("🔍 Recent Traffic")
    st.dataframe(df.tail(10).sort_values(by='Timestamp', ascending=False))