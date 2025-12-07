"""
EC2 Metrics Collection Server
Deploy this on EC2 instances to expose real-time metrics
"""
from flask import Flask, jsonify
import psutil
import time
import socket

app = Flask(__name__)

def get_network_io():
    """Get network I/O statistics in Mbps"""
    net_io_start = psutil.net_io_counters()
    time.sleep(1)  # Measure over 1 second
    net_io_end = psutil.net_io_counters()
    
    # Calculate Mbps (bytes to megabits)
    bytes_sent = net_io_end.bytes_sent - net_io_start.bytes_sent
    bytes_recv = net_io_end.bytes_recv - net_io_start.bytes_recv
    
    network_out_mbps = (bytes_sent * 8) / (1024 * 1024)  # Convert to Mbps
    network_in_mbps = (bytes_recv * 8) / (1024 * 1024)
    
    return network_in_mbps, network_out_mbps

def get_response_time():
    """Simulate response time measurement (customize based on your app)"""
    # For a web server, you could measure actual request response times
    # For now, using a simple latency check
    start = time.time()
    try:
        # Simple ping to localhost
        socket.gethostbyname(socket.gethostname())
    except:
        pass
    end = time.time()
    return (end - start) * 1000  # Convert to milliseconds

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return all 6 metrics needed for the LSTM model"""
    try:
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory utilization
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Get network I/O
        network_in_mbps, network_out_mbps = get_network_io()
        
        # Get response time
        response_time_ms = get_response_time()
        
        # Get instance metadata (if available on EC2)
        instance_id = "local"
        instance_type = "unknown"
        try:
            import requests
            # Try to get EC2 metadata
            metadata_url = "http://169.254.169.254/latest/meta-data/"
            instance_id = requests.get(f"{metadata_url}instance-id", timeout=0.5).text
            instance_type = requests.get(f"{metadata_url}instance-type", timeout=0.5).text
        except:
            pass
        
        metrics = {
            'instance_id': instance_id,
            'instance_type': instance_type,
            'timestamp': time.time(),
            'metrics': {
                'Network_In_Mbps': round(network_in_mbps, 2),
                'Network_Out_Mbps': round(network_out_mbps, 2),
                'Response_Time_ms': round(response_time_ms, 2),
                'CPU_Utilization_Percent': round(cpu_percent, 2),
                'Memory_Utilization_Percent': round(memory_percent, 2),
                'Disk_Usage_Percent': round(disk_percent, 2)
            }
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'metrics-server'}), 200

@app.route('/', methods=['GET'])
def index():
    """Display metrics in HTML format"""
    try:
        # Get CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory utilization
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Get network I/O
        network_in_mbps, network_out_mbps = get_network_io()
        
        # Get response time
        response_time_ms = get_response_time()
        
        # Get instance metadata
        instance_id = "local"
        instance_type = "unknown"
        try:
            import requests
            metadata_url = "http://169.254.169.254/latest/meta-data/"
            instance_id = requests.get(f"{metadata_url}instance-id", timeout=0.5).text
            instance_type = requests.get(f"{metadata_url}instance-type", timeout=0.5).text
        except:
            pass
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EC2 Metrics Server</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .container {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    padding: 15px;
                    margin: 10px 0;
                    background: #f9f9f9;
                    border-left: 4px solid #4CAF50;
                    border-radius: 5px;
                }}
                .metric-name {{
                    font-weight: bold;
                    color: #555;
                }}
                .metric-value {{
                    color: #4CAF50;
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                .instance-info {{
                    background: #e8f5e9;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .refresh-note {{
                    text-align: center;
                    color: #888;
                    margin-top: 20px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>EC2 Instance Metrics</h1>
                
                <div class="instance-info">
                    <div><strong>Instance ID:</strong> {instance_id}</div>
                    <div><strong>Instance Type:</strong> {instance_type}</div>
                    <div><strong>Timestamp:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
                
                <div class="metric">
                    <span class="metric-name">CPU Utilization</span>
                    <span class="metric-value">{cpu_percent:.2f}%</span>
                </div>
                
                <div class="metric">
                    <span class="metric-name">Memory Utilization</span>
                    <span class="metric-value">{memory_percent:.2f}%</span>
                </div>
                
                <div class="metric">
                    <span class="metric-name">Disk Usage</span>
                    <span class="metric-value">{disk_percent:.2f}%</span>
                </div>
                
                <div class="metric">
                    <span class="metric-name">Network In</span>
                    <span class="metric-value">{network_in_mbps:.2f} Mbps</span>
                </div>
                
                <div class="metric">
                    <span class="metric-name">Network Out</span>
                    <span class="metric-value">{network_out_mbps:.2f} Mbps</span>
                </div>
                
                <div class="metric">
                    <span class="metric-name">Response Time</span>
                    <span class="metric-value">{response_time_ms:.2f} ms</span>
                </div>
                
                <div class="refresh-note">
                    Page auto-refreshes every 5 seconds | 
                    <a href="/metrics">JSON API</a> | 
                    <a href="/health">Health Check</a>
                </div>
            </div>
        </body>
        </html>
        """
        return html
        
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p>", 500

if __name__ == '__main__':
    print("="*60)
    print("EC2 Metrics Collection Server")
    print("="*60)
    print("Endpoints:")
    print("  GET /           - HTML metrics dashboard (auto-refresh)")
    print("  GET /metrics    - JSON metrics API")
    print("  GET /health     - Health check")
    print("="*60)
    print("\nServer starting on http://0.0.0.0:8080")
    print("="*60)
    
    app.run(host='0.0.0.0', port=8080, debug=False)
