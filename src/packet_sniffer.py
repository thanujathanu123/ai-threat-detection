# src/packet_sniffer.py

import pandas as pd
import numpy as np
import time
import threading
import queue
import os
import json
from scapy.all import sniff, IP, TCP, UDP, ICMP
from datetime import datetime

class PacketSniffer:
    def __init__(self, packet_queue, interface=None, max_packets=100):
        """
        Initialize the packet sniffer
        
        Args:
            packet_queue: Queue to store captured packets
            interface: Network interface to sniff on (None for all interfaces)
            max_packets: Maximum number of packets to store in memory
        """
        self.packet_queue = packet_queue
        self.interface = interface
        self.max_packets = max_packets
        self.stop_sniffing = threading.Event()
        self.packet_count = 0
        self.packet_buffer = []
        self.lock = threading.Lock()
        
    def _process_packet(self, packet):
        """Process a single packet and extract relevant features"""
        packet_data = {}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # Basic packet info
        packet_data["Timestamp"] = timestamp
        packet_data["Protocol"] = "OTHER"
        
        # Extract IP layer information if present
        if IP in packet:
            packet_data["Source IP"] = packet[IP].src
            packet_data["Destination IP"] = packet[IP].dst
            packet_data["TTL"] = packet[IP].ttl
            packet_data["Length"] = len(packet)
            
            # TCP specific features
            if TCP in packet:
                packet_data["Protocol"] = "TCP"
                packet_data["Source Port"] = packet[TCP].sport
                packet_data["Destination Port"] = packet[TCP].dport
                packet_data["TCP Flags"] = packet[TCP].flags
                packet_data["Window Size"] = packet[TCP].window
                
            # UDP specific features
            elif UDP in packet:
                packet_data["Protocol"] = "UDP"
                packet_data["Source Port"] = packet[UDP].sport
                packet_data["Destination Port"] = packet[UDP].dport
                packet_data["Length"] = packet[UDP].len
                
            # ICMP specific features
            elif ICMP in packet:
                packet_data["Protocol"] = "ICMP"
                packet_data["ICMP Type"] = packet[ICMP].type
                packet_data["ICMP Code"] = packet[ICMP].code
        
        # Add placeholder for prediction
        packet_data["Label"] = 0  # Default to normal traffic
        
        with self.lock:
            self.packet_buffer.append(packet_data)
            self.packet_count += 1
            
            # If buffer is full, convert to DataFrame and put in queue
            if len(self.packet_buffer) >= self.max_packets:
                self._buffer_to_queue()
                
        return packet_data
    
    def _buffer_to_queue(self):
        """Convert buffer to DataFrame and put in queue"""
        if not self.packet_buffer:
            return
            
        with self.lock:
            df = pd.DataFrame(self.packet_buffer)
            self.packet_buffer = []
            
        # Put DataFrame in queue for processing
        try:
            self.packet_queue.put(df, block=False)
        except queue.Full:
            # If queue is full, just drop the packets
            pass
    
    def _packet_callback(self, packet):
        """Callback function for each captured packet"""
        if self.stop_sniffing.is_set():
            return True  # Stop sniffing
            
        self._process_packet(packet)
        return None  # Continue sniffing
    
    def start_sniffing(self):
        """Start sniffing packets in a separate thread"""
        sniff_thread = threading.Thread(
            target=sniff,
            kwargs={
                "prn": self._packet_callback,
                "store": False,
                "iface": self.interface,
                "stop_filter": lambda p: self.stop_sniffing.is_set()
            }
        )
        sniff_thread.daemon = True
        sniff_thread.start()
        return sniff_thread
    
    def stop(self):
        """Stop sniffing packets"""
        self.stop_sniffing.set()
        # Flush remaining packets to queue
        self._buffer_to_queue()

class PacketProcessor:
    def __init__(self, packet_queue, result_queue, model, preprocess_func):
        """
        Initialize the packet processor
        
        Args:
            packet_queue: Queue to get captured packets from
            result_queue: Queue to put processed results into
            model: ML model for prediction
            preprocess_func: Function to preprocess data before prediction
        """
        self.packet_queue = packet_queue
        self.result_queue = result_queue
        self.model = model
        self.preprocess_func = preprocess_func
        self.stop_processing = threading.Event()
        
    def process_packets(self):
        """Process packets from queue and make predictions"""
        while not self.stop_processing.is_set():
            try:
                # Get DataFrame from queue with timeout
                df = self.packet_queue.get(timeout=1)
                
                # Skip if DataFrame is empty
                if df.empty:
                    continue
                    
                # Make a copy of the original data
                original_df = df.copy()
                
                # Preprocess data for model
                try:
                    # Add dummy Label column if not present (required by preprocess_func)
                    if 'Label' not in df.columns:
                        df['Label'] = 0
                        
                    X, _ = self.preprocess_func(df)
                    
                    # Make predictions
                    y_pred = self.model.predict(X)
                    y_prob = self.model.predict_proba(X)
                    
                    # Add predictions to original DataFrame
                    original_df['Prediction'] = y_pred
                    original_df['Confidence (%)'] = np.max(y_prob, axis=1) * 100
                    
                    # Put results in result queue
                    self.result_queue.put(original_df)
                    
                except Exception as e:
                    print(f"Error processing packets: {e}")
                    
            except queue.Empty:
                # Queue is empty, just continue
                continue
                
    def start_processing(self):
        """Start processing packets in a separate thread"""
        process_thread = threading.Thread(target=self.process_packets)
        process_thread.daemon = True
        process_thread.start()
        return process_thread
        
    def stop(self):
        """Stop processing packets"""
        self.stop_processing.set()

def start_packet_capture(interface=None, model=None, preprocess_func=None):
    """
    Start packet capture and processing
    
    Args:
        interface: Network interface to sniff on
        model: ML model for prediction
        preprocess_func: Function to preprocess data
        
    Returns:
        packet_sniffer: PacketSniffer instance
        packet_processor: PacketProcessor instance
        result_queue: Queue with processed results
    """
    # Create queues for communication between threads
    packet_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
    result_queue = queue.Queue()
    
    # Create and start packet sniffer
    packet_sniffer = PacketSniffer(packet_queue, interface)
    sniffer_thread = packet_sniffer.start_sniffing()
    
    # Create and start packet processor if model is provided
    packet_processor = None
    if model is not None and preprocess_func is not None:
        packet_processor = PacketProcessor(packet_queue, result_queue, model, preprocess_func)
        processor_thread = packet_processor.start_processing()
    
    return packet_sniffer, packet_processor, result_queue

if __name__ == "__main__":
    # Simple test to capture packets
    packet_queue = queue.Queue()
    sniffer = PacketSniffer(packet_queue)
    thread = sniffer.start_sniffing()
    
    try:
        print("Sniffing packets... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sniffer.stop()
        print(f"Captured {sniffer.packet_count} packets")