# run_monitor.py

import argparse
import time
import pandas as pd
from data.predict import load_model
from data.data_preprocessing import preprocess_data
from src.packet_sniffer import start_packet_capture

def main():
    """
    Command-line tool to run network monitoring without the Streamlit dashboard
    """
    parser = argparse.ArgumentParser(description="AI-Powered Network Threat Detection")
    parser.add_argument("--interface", "-i", type=str, default=None,
                        help="Network interface to monitor (default: all interfaces)")
    parser.add_argument("--threshold", "-t", type=float, default=85.0,
                        help="Confidence threshold for alerts (default: 85.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file for detected threats (default: None)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed information about each packet")
    
    args = parser.parse_args()
    
    print(f"Loading model...")
    model = load_model()
    print(f"Starting packet capture on interface: {args.interface or 'all interfaces'}")
    
    # Start packet capture
    sniffer, processor, result_queue = start_packet_capture(
        interface=args.interface,
        model=model,
        preprocess_func=preprocess_data
    )
    
    # Track detected threats
    all_threats = []
    
    try:
        print("Monitoring network traffic... Press Ctrl+C to stop")
        while True:
            # Get results from queue
            try:
                df = result_queue.get(timeout=1)
                result_queue.task_done()
                
                # Filter threats based on threshold
                threats = df[(df['Prediction'] == 1) & (df['Confidence (%)'] >= args.threshold)]
                
                if not threats.empty:
                    # Store threats for later output
                    all_threats.append(threats)
                    
                    # Print threat information
                    print(f"\n[!] Detected {len(threats)} suspicious connections!")
                    
                    if args.verbose:
                        for _, row in threats.iterrows():
                            print(f"  - {row.get('Source IP', 'Unknown')}:{row.get('Source Port', 'Unknown')} -> "
                                  f"{row.get('Destination IP', 'Unknown')}:{row.get('Destination Port', 'Unknown')} "
                                  f"({row.get('Protocol', 'Unknown')}) - "
                                  f"Confidence: {row.get('Confidence (%)', 0):.2f}%")
            
            except Exception:
                # No results in queue, just continue
                pass
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        sniffer.stop()
        processor.stop()
        
        # Save detected threats to CSV if output file specified
        if args.output and all_threats:
            threats_df = pd.concat(all_threats, ignore_index=True)
            threats_df.to_csv(args.output, index=False)
            print(f"Saved {len(threats_df)} detected threats to {args.output}")
            
        print(f"Captured {sniffer.packet_count} packets")
        print("Done!")

if __name__ == "__main__":
    main()