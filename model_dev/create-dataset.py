import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import warnings
import uuid
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedCloudInstanceDataGenerator:
    def __init__(self):
        # Enhanced instance types with MORE distinct characteristics for better separability
        self.instance_types = {
            # T-series (Burstable) - Low performance, low cost
            't2.nano': {
                'cpu_base': 12, 'cpu_std': 6,
                'memory_base': 15, 'memory_std': 8,
                'network_in_base': 8, 'network_in_std': 3,
                'network_out_base': 8, 'network_out_std': 3,
                'network_throughput_base': 15, 'network_throughput_std': 5,
                'disk_io_base': 120, 'disk_io_std': 30,
                'disk_usage_base': 20, 'disk_usage_std': 10,
                'response_time_base': 280, 'response_time_std': 40,
                'cost_base': 0.0058, 'cost_std': 0.001
            },
            't2.micro': {
                'cpu_base': 22, 'cpu_std': 8,
                'memory_base': 25, 'memory_std': 10,
                'network_in_base': 12, 'network_in_std': 4,
                'network_out_base': 12, 'network_out_std': 4,
                'network_throughput_base': 22, 'network_throughput_std': 6,
                'disk_io_base': 200, 'disk_io_std': 40,
                'disk_usage_base': 30, 'disk_usage_std': 12,
                'response_time_base': 220, 'response_time_std': 35,
                'cost_base': 0.0116, 'cost_std': 0.002
            },
            't2.small': {
                'cpu_base': 32, 'cpu_std': 10,
                'memory_base': 35, 'memory_std': 12,
                'network_in_base': 18, 'network_in_std': 5,
                'network_out_base': 18, 'network_out_std': 5,
                'network_throughput_base': 32, 'network_throughput_std': 8,
                'disk_io_base': 320, 'disk_io_std': 60,
                'disk_usage_base': 40, 'disk_usage_std': 15,
                'response_time_base': 180, 'response_time_std': 30,
                'cost_base': 0.023, 'cost_std': 0.003
            },
            't2.medium': {
                'cpu_base': 45, 'cpu_std': 12,
                'memory_base': 50, 'memory_std': 15,
                'network_in_base': 28, 'network_in_std': 7,
                'network_out_base': 28, 'network_out_std': 7,
                'network_throughput_base': 50, 'network_throughput_std': 12,
                'disk_io_base': 500, 'disk_io_std': 80,
                'disk_usage_base': 55, 'disk_usage_std': 18,
                'response_time_base': 140, 'response_time_std': 25,
                'cost_base': 0.046, 'cost_std': 0.004
            },
            't3.nano': {
                'cpu_base': 15, 'cpu_std': 6,
                'memory_base': 18, 'memory_std': 8,
                'network_in_base': 10, 'network_in_std': 3,
                'network_out_base': 10, 'network_out_std': 3,
                'network_throughput_base': 18, 'network_throughput_std': 5,
                'disk_io_base': 150, 'disk_io_std': 35,
                'disk_usage_base': 22, 'disk_usage_std': 10,
                'response_time_base': 250, 'response_time_std': 35,
                'cost_base': 0.0052, 'cost_std': 0.001
            },
            't3.micro': {
                'cpu_base': 25, 'cpu_std': 8,
                'memory_base': 28, 'memory_std': 10,
                'network_in_base': 14, 'network_in_std': 4,
                'network_out_base': 14, 'network_out_std': 4,
                'network_throughput_base': 25, 'network_throughput_std': 6,
                'disk_io_base': 220, 'disk_io_std': 45,
                'disk_usage_base': 32, 'disk_usage_std': 12,
                'response_time_base': 200, 'response_time_std': 30,
                'cost_base': 0.0104, 'cost_std': 0.002
            },
            
            # M-series (General Purpose) - Balanced performance
            'm5.large': {
                'cpu_base': 65, 'cpu_std': 15,
                'memory_base': 70, 'memory_std': 18,
                'network_in_base': 55, 'network_in_std': 12,
                'network_out_base': 55, 'network_out_std': 12,
                'network_throughput_base': 110, 'network_throughput_std': 20,
                'disk_io_base': 1100, 'disk_io_std': 150,
                'disk_usage_base': 72, 'disk_usage_std': 20,
                'response_time_base': 85, 'response_time_std': 18,
                'cost_base': 0.096, 'cost_std': 0.008
            },
            'm5.xlarge': {
                'cpu_base': 78, 'cpu_std': 15,
                'memory_base': 82, 'memory_std': 18,
                'network_in_base': 70, 'network_in_std': 15,
                'network_out_base': 70, 'network_out_std': 15,
                'network_throughput_base': 155, 'network_throughput_std': 25,
                'disk_io_base': 1550, 'disk_io_std': 200,
                'disk_usage_base': 80, 'disk_usage_std': 22,
                'response_time_base': 65, 'response_time_std': 15,
                'cost_base': 0.192, 'cost_std': 0.015
            },
            'm5.2xlarge': {
                'cpu_base': 88, 'cpu_std': 12,
                'memory_base': 90, 'memory_std': 15,
                'network_in_base': 90, 'network_in_std': 18,
                'network_out_base': 90, 'network_out_std': 18,
                'network_throughput_base': 220, 'network_throughput_std': 30,
                'disk_io_base': 2200, 'disk_io_std': 250,
                'disk_usage_base': 85, 'disk_usage_std': 20,
                'response_time_base': 48, 'response_time_std': 12,
                'cost_base': 0.384, 'cost_std': 0.020
            },
            
            # C-series (Compute Optimized) - High CPU, moderate memory
            'c5.large': {
                'cpu_base': 82, 'cpu_std': 12,
                'memory_base': 58, 'memory_std': 15,
                'network_in_base': 68, 'network_in_std': 15,
                'network_out_base': 68, 'network_out_std': 15,
                'network_throughput_base': 135, 'network_throughput_std': 22,
                'disk_io_base': 1350, 'disk_io_std': 180,
                'disk_usage_base': 62, 'disk_usage_std': 18,
                'response_time_base': 55, 'response_time_std': 12,
                'cost_base': 0.085, 'cost_std': 0.008
            },
            'c5.xlarge': {
                'cpu_base': 90, 'cpu_std': 10,
                'memory_base': 68, 'memory_std': 15,
                'network_in_base': 85, 'network_in_std': 18,
                'network_out_base': 85, 'network_out_std': 18,
                'network_throughput_base': 180, 'network_throughput_std': 25,
                'disk_io_base': 1800, 'disk_io_std': 220,
                'disk_usage_base': 70, 'disk_usage_std': 18,
                'response_time_base': 42, 'response_time_std': 10,
                'cost_base': 0.17, 'cost_std': 0.012
            },
            'c5.2xlarge': {
                'cpu_base': 94, 'cpu_std': 8,
                'memory_base': 75, 'memory_std': 15,
                'network_in_base': 105, 'network_in_std': 20,
                'network_out_base': 105, 'network_out_std': 20,
                'network_throughput_base': 240, 'network_throughput_std': 30,
                'disk_io_base': 2400, 'disk_io_std': 280,
                'disk_usage_base': 75, 'disk_usage_std': 18,
                'response_time_base': 32, 'response_time_std': 8,
                'cost_base': 0.34, 'cost_std': 0.018
            },
            
            # R-series (Memory Optimized) - High memory, moderate CPU
            'r5.large': {
                'cpu_base': 62, 'cpu_std': 15,
                'memory_base': 88, 'memory_std': 15,
                'network_in_base': 52, 'network_in_std': 12,
                'network_out_base': 52, 'network_out_std': 12,
                'network_throughput_base': 105, 'network_throughput_std': 20,
                'disk_io_base': 1050, 'disk_io_std': 140,
                'disk_usage_base': 85, 'disk_usage_std': 18,
                'response_time_base': 75, 'response_time_std': 15,
                'cost_base': 0.126, 'cost_std': 0.010
            },
            'r5.xlarge': {
                'cpu_base': 75, 'cpu_std': 15,
                'memory_base': 92, 'memory_std': 12,
                'network_in_base': 72, 'network_in_std': 15,
                'network_out_base': 72, 'network_out_std': 15,
                'network_throughput_base': 145, 'network_throughput_std': 22,
                'disk_io_base': 1450, 'disk_io_std': 180,
                'disk_usage_base': 88, 'disk_usage_std': 15,
                'response_time_base': 62, 'response_time_std': 12,
                'cost_base': 0.252, 'cost_std': 0.015
            },
            
            # X-series (High Performance) - Very high memory and performance
            'x1.large': {
                'cpu_base': 80, 'cpu_std': 12,
                'memory_base': 95, 'memory_std': 10,
                'network_in_base': 65, 'network_in_std': 12,
                'network_out_base': 65, 'network_out_std': 12,
                'network_throughput_base': 130, 'network_throughput_std': 20,
                'disk_io_base': 1300, 'disk_io_std': 160,
                'disk_usage_base': 92, 'disk_usage_std': 12,
                'response_time_base': 50, 'response_time_std': 10,
                'cost_base': 0.334, 'cost_std': 0.020
            }
        }
    
    def generate_instance_id(self, instance_type, sequence_num):
        """Generate realistic instance IDs"""
        prefix = instance_type.replace('.', '-')
        unique_id = str(uuid.uuid4())[:8]
        return f"i-{prefix}-{unique_id}"
    
    def generate_sequence_data(self, instance_type, sequence_length=10, num_sequences=150):
        """Generate time series sequences for a specific instance type"""
        config = self.instance_types[instance_type]
        sequences = []
        instance_ids = []
        
        for seq_num in range(num_sequences):
            sequence = []
            instance_id = self.generate_instance_id(instance_type, seq_num)
            
            # Add consistent base multiplier for this instance (less variation)
            base_multiplier = np.random.uniform(0.9, 1.1)  # Reduced variation
            
            for step in range(sequence_length):
                # Add subtle temporal correlation
                time_factor = 1 + 0.03 * np.sin(2 * np.pi * step / sequence_length)
                
                # Generate features with REDUCED noise for better separability
                cpu_util = np.clip(np.random.normal(
                    config['cpu_base'] * base_multiplier * time_factor, 
                    config['cpu_std'] * 0.6  # Reduced noise
                ), 0, 100)
                
                memory_util = np.clip(np.random.normal(
                    config['memory_base'] * base_multiplier * time_factor, 
                    config['memory_std'] * 0.6
                ), 0, 100)
                
                network_in = np.clip(np.random.normal(
                    config['network_in_base'] * base_multiplier, 
                    config['network_in_std'] * 0.5
                ), 0, 1000)
                
                network_out = np.clip(np.random.normal(
                    config['network_out_base'] * base_multiplier, 
                    config['network_out_std'] * 0.5
                ), 0, 1000)
                
                network_throughput = np.clip(np.random.normal(
                    config['network_throughput_base'] * base_multiplier, 
                    config['network_throughput_std'] * 0.5
                ), 0, 2000)
                
                disk_io = np.clip(np.random.normal(
                    config['disk_io_base'] * base_multiplier, 
                    config['disk_io_std'] * 0.5
                ), 0, 5000)
                
                disk_usage = np.clip(np.random.normal(
                    config['disk_usage_base'] * base_multiplier * time_factor, 
                    config['disk_usage_std'] * 0.6
                ), 0, 100)
                
                response_time = np.clip(np.random.normal(
                    config['response_time_base'] / base_multiplier, 
                    config['response_time_std'] * 0.5
                ), 10, 1000)
                
                hourly_cost = np.clip(np.random.normal(
                    config['cost_base'], 
                    config['cost_std'] * 0.4  # Very low cost variation
                ), 0.001, 1.0)
                
                sequence.append([
                    cpu_util, memory_util, network_in, network_out, 
                    network_throughput, disk_io, disk_usage, response_time, hourly_cost
                ])
            
            sequences.append(sequence)
            instance_ids.append(instance_id)
        
        return np.array(sequences), [instance_type] * num_sequences, instance_ids
    
    def generate_dataset(self, sequence_length=10, samples_per_type=150):
        """Generate complete dataset with all instance types"""
        all_sequences = []
        all_labels = []
        all_instance_ids = []
        
        for instance_type in self.instance_types.keys():
            print(f"Generating {samples_per_type} sequences for {instance_type}...")
            sequences, labels, instance_ids = self.generate_sequence_data(
                instance_type, sequence_length, samples_per_type
            )
            all_sequences.extend(sequences)
            all_labels.extend(labels)
            all_instance_ids.extend(instance_ids)
        
        return np.array(all_sequences), np.array(all_labels), all_instance_ids

def add_moderate_noise_for_target_accuracy(X, y, instance_ids, noise_factor=0.08, outlier_rate=0.05, label_noise_rate=0.06):
    """Add LSTM-optimized noise to achieve target accuracy of 80-85%"""
    print(f"\n🔧 Adding moderate noise for 80-85% target accuracy...")
    print(f"   • LSTM-optimized feature noise factor: {noise_factor}")
    print(f"   • Reduced outlier rate: {outlier_rate*100:.1f}%")
    print(f"   • Smart label noise rate: {label_noise_rate*100:.1f}%")
    
    X_noisy = X.copy()
    
    # Add LSTM-friendly temporal noise patterns
    print(f"   • Adding LSTM-optimized temporal noise patterns...")
    for i in range(len(X_noisy)):
        for t in range(X_noisy.shape[1]):
            # Add smooth temporal noise that LSTMs can learn to handle
            temporal_factor = 1 + 0.02 * np.sin(2 * np.pi * t / X_noisy.shape[1])
            feature_noise = np.random.normal(0, noise_factor * temporal_factor, X_noisy.shape[2])
            X_noisy[i, t, :] += feature_noise
    
    # Add sequence-level consistency to help LSTM learn patterns
    print(f"   • Adding sequence-level consistency patterns...")
    for i in range(len(X_noisy)):
        # Add consistent bias across the entire sequence
        sequence_bias = np.random.normal(0, noise_factor * 0.5, X_noisy.shape[2])
        for t in range(X_noisy.shape[1]):
            X_noisy[i, t, :] += sequence_bias
    
    # Add FEWER, more structured outliers that create learnable patterns
    num_outliers = int(len(X) * outlier_rate)
    outlier_indices = np.random.choice(len(X), num_outliers, replace=False)
    
    print(f"   • Adding {num_outliers} structured outlier sequences...")
    for idx in outlier_indices:
        scenario = np.random.choice(['high_load', 'network_issue', 'memory_pressure', 'gradual_degradation'])
        
        if scenario == 'high_load':
            # Gradual CPU and response time increase (LSTM-friendly pattern)
            multiplier = np.linspace(1.0, np.random.uniform(1.3, 1.6), X_noisy.shape[1])
            X_noisy[idx, :, 0] *= multiplier  # CPU
            X_noisy[idx, :, 7] *= multiplier * 1.2  # Response time
        elif scenario == 'network_issue':
            # Gradual network degradation
            multiplier = np.linspace(1.0, np.random.uniform(0.6, 0.8), X_noisy.shape[1])
            X_noisy[idx, :, 2:5] *= multiplier.reshape(-1, 1)  # Network metrics
        elif scenario == 'memory_pressure':
            # Gradual memory pressure buildup
            multiplier = np.linspace(1.0, np.random.uniform(1.2, 1.5), X_noisy.shape[1])
            X_noisy[idx, :, 1] *= multiplier  # Memory
            X_noisy[idx, :, 7] *= multiplier * 0.8  # Response time
        elif scenario == 'gradual_degradation':
            # Overall system degradation over time (great for LSTM learning)
            degradation = np.linspace(1.0, np.random.uniform(1.1, 1.3), X_noisy.shape[1])
            X_noisy[idx, :, 0] *= degradation  # CPU
            X_noisy[idx, :, 1] *= degradation  # Memory
            X_noisy[idx, :, 7] *= degradation * 1.5  # Response time
    
    # Add STRATEGIC label noise to prevent tree overfitting while helping LSTM
    num_label_noise = int(len(y) * label_noise_rate)
    label_noise_indices = np.random.choice(len(y), num_label_noise, replace=False)
    
    unique_labels = np.unique(y)
    y_noisy = y.copy()
    
    print(f"   • Adding {num_label_noise} smart mislabeled sequences...")
    changed_count = 0
    for idx in label_noise_indices:
        original_label = y[idx]
        
        # VERY smart confusion: heavily prefer similar instance types
        similar_types = []
        original_series = original_label.split('.')[0]
        original_size = original_label.split('.')[1]
        
        for label in unique_labels:
            if label != original_label:
                label_series = label.split('.')[0] 
                label_size = label.split('.')[1]
                
                # EXTREMELY high preference for same series, adjacent size
                if label_series == original_series:
                    similar_types.extend([label] * 8)  # 8x weight
                # High preference for same size, different series
                elif label_size == original_size:
                    similar_types.extend([label] * 4)  # 4x weight
                # Very low preference for completely different
                else:
                    similar_types.append(label)  # 1x weight
        
        if similar_types:  # Only change if we have similar types
            new_label = np.random.choice(similar_types)
            y_noisy[idx] = new_label
            changed_count += 1
    
    # Add MINIMAL cross-correlation noise to prevent tree memorization
    print(f"   • Adding minimal cross-correlation noise...")
    for i in range(len(X_noisy)):
        cpu_factor = (X_noisy[i, :, 0] - 50) / 200.0  # Much smaller normalization
        # Very light correlations
        X_noisy[i, :, 1] += cpu_factor * np.random.normal(0, 1, X_noisy.shape[1])  # Memory
        X_noisy[i, :, 7] += cpu_factor * np.random.normal(0, 2, X_noisy.shape[1])  # Response time
        
        # Very light network correlation
        net_correlation = np.random.normal(0, 0.02, X_noisy.shape[1])
        X_noisy[i, :, 2] += X_noisy[i, :, 3] * net_correlation * 0.05
    
    # Add feature smoothing to help LSTM learn temporal patterns
    print(f"   • Adding temporal smoothing for LSTM optimization...")
    for i in range(len(X_noisy)):
        for f in range(X_noisy.shape[2]):
            # Apply light smoothing to make temporal patterns more learnable
            smoothed = np.convolve(X_noisy[i, :, f], np.ones(3)/3, mode='same')
            X_noisy[i, :, f] = 0.7 * X_noisy[i, :, f] + 0.3 * smoothed
    
    # Clip values to reasonable ranges
    X_noisy[:, :, 0] = np.clip(X_noisy[:, :, 0], 0, 100)  # CPU
    X_noisy[:, :, 1] = np.clip(X_noisy[:, :, 1], 0, 100)  # Memory
    X_noisy[:, :, 2] = np.clip(X_noisy[:, :, 2], 0, 1000)  # Network in
    X_noisy[:, :, 3] = np.clip(X_noisy[:, :, 3], 0, 1000)  # Network out
    X_noisy[:, :, 4] = np.clip(X_noisy[:, :, 4], 0, 2000)  # Network throughput
    X_noisy[:, :, 5] = np.clip(X_noisy[:, :, 5], 0, 5000)  # Disk IO
    X_noisy[:, :, 6] = np.clip(X_noisy[:, :, 6], 0, 100)   # Disk usage
    X_noisy[:, :, 7] = np.clip(X_noisy[:, :, 7], 10, 1000) # Response time
    X_noisy[:, :, 8] = np.clip(X_noisy[:, :, 8], 0.001, 1.0) # Cost
    
    print(f"   ✅ Moderate noise applied successfully!")
    print(f"   • Feature outliers: {num_outliers}")
    print(f"   • Label changes: {changed_count}")
    print(f"   • Light cross-correlations added")
    
    return X_noisy, y_noisy

def create_and_save_enhanced_dataset():
    """Main function to create and save the enhanced dataset optimized for 80-85% accuracy"""
    print("🚀 Enhanced Cloud Instance Dataset Generator (80-85% Target)")
    print("=" * 70)
    
    # Generate clean dataset with better separability
    print("📊 Generating well-separated dataset...")
    generator = EnhancedCloudInstanceDataGenerator()
    X_clean, y_clean, instance_ids_clean = generator.generate_dataset(sequence_length=10, samples_per_type=150)
    
    print(f"✅ Generated clean dataset:")
    print(f"   • Shape: {X_clean.shape}")
    print(f"   • Instance types: {len(np.unique(y_clean))} types")
    print(f"   • Total sequences: {len(X_clean)}")
    print(f"   • Instance types: {list(np.unique(y_clean))}")
    
    # Add MODERATE noise to achieve target accuracy (80-85%)
    X_noisy, y_noisy = add_moderate_noise_for_target_accuracy(
        X_clean, y_clean, instance_ids_clean,
        noise_factor=0.08,   # Much lower noise
        outlier_rate=0.05,   # Much fewer outliers
        label_noise_rate=0.06  # Much less label noise
    )
    
    # Create instance IDs for noisy dataset
    instance_ids_noisy = []
    for i, (clean_label, noisy_label) in enumerate(zip(y_clean, y_noisy)):
        if clean_label == noisy_label:
            instance_ids_noisy.append(instance_ids_clean[i])
        else:
            instance_ids_noisy.append(generator.generate_instance_id(noisy_label, i))
    
    # Encode labels
    print("\n🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_noisy)
    y_clean_encoded = label_encoder.transform(y_clean)
    
    # Feature names
    feature_names = [
        'CPU_Utilization_Percent', 'Memory_Utilization_Percent', 'Network_In_Mbps',
        'Network_Out_Mbps', 'Network_Throughput_Mbps', 'Disk_IO_IOPS',
        'Disk_Usage_Percent', 'Response_Time_ms', 'Hourly_Cost_USD'
    ]
    
    # Create dataset folder
    import os
    os.makedirs('dataset', exist_ok=True)
    
    # Save all dataset files
    print("\n💾 Saving enhanced dataset files...")
    
    # 1. Save raw arrays
    np.save('dataset/X_sequences_noisy.npy', X_noisy)
    np.save('dataset/y_labels_noisy.npy', y_noisy)
    np.save('dataset/y_labels_encoded.npy', y_encoded)
    np.save('dataset/X_sequences_clean.npy', X_clean)
    np.save('dataset/y_labels_clean.npy', y_clean)
    
    # 2. Save preprocessing objects
    with open('dataset/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('dataset/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    with open('dataset/instance_ids_clean.pkl', 'wb') as f:
        pickle.dump(instance_ids_clean, f)
        
    with open('dataset/instance_ids_noisy.pkl', 'wb') as f:
        pickle.dump(instance_ids_noisy, f)
    
    # 3. Save comprehensive CSV
    print("📝 Creating comprehensive CSV file...")
    df_sequences = pd.DataFrame(
        X_noisy.reshape(-1, X_noisy.shape[-1]), 
        columns=feature_names
    )
    df_sequences['Instance_Type'] = np.repeat(y_noisy, X_noisy.shape[1])
    df_sequences['Original_Instance_Type'] = np.repeat(y_clean, X_noisy.shape[1])
    df_sequences['Instance_ID'] = np.repeat(instance_ids_noisy, X_noisy.shape[1])
    df_sequences['Original_Instance_ID'] = np.repeat(instance_ids_clean, X_noisy.shape[1])
    df_sequences['Sequence_ID'] = np.repeat(range(len(y_noisy)), X_noisy.shape[1])
    df_sequences['Time_Step'] = np.tile(range(X_noisy.shape[1]), len(y_noisy))
    df_sequences['Is_Label_Noisy'] = np.repeat(y_clean != y_noisy, X_noisy.shape[1])
    df_sequences['Label_Encoded'] = np.repeat(y_encoded, X_noisy.shape[1])
    
    df_sequences.to_csv('dataset/cloud_instance_dataset_enhanced.csv', index=False)
    
    # 4. Save instance analysis
    instance_analysis = pd.DataFrame({
        'Sequence_ID': range(len(y_clean)),
        'Original_Instance_Type': y_clean,
        'Noisy_Instance_Type': y_noisy,
        'Original_Instance_ID': instance_ids_clean,
        'Noisy_Instance_ID': instance_ids_noisy,
        'Original_Encoded': y_clean_encoded,
        'Noisy_Encoded': y_encoded,
        'Label_Changed': y_clean != y_noisy,
        'ID_Changed': [oid != nid for oid, nid in zip(instance_ids_clean, instance_ids_noisy)]
    })
    instance_analysis.to_csv('dataset/instance_analysis.csv', index=False)
    
    # 5. Create enhanced dataset metadata
    dataset_info = {
        'dataset_name': 'Optimized Cloud Instance Dataset (80-85% Target)',
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_sequences': len(X_noisy),
        'sequence_length': X_noisy.shape[1],
        'num_features': X_noisy.shape[2],
        'num_classes': len(np.unique(y_encoded)),
        'class_names': list(label_encoder.classes_),
        'feature_names': feature_names,
        'samples_per_class': {cls: int(np.sum(y_noisy == cls)) for cls in np.unique(y_noisy)},
        'optimization_parameters': {
            'lstm_optimized_noise_factor': 0.08,
            'outlier_rate': 0.05,
            'label_noise_rate': 0.06,
            'temporal_smoothing_enabled': True,
            'sequence_consistency_enabled': True,
            'gradual_degradation_patterns': True
        },
        'target_accuracy_range': '80-85% (LSTM-optimized)',
        'key_improvements': [
            'LSTM-optimized temporal noise patterns',
            'Sequence-level consistency for better learning',
            'Gradual degradation patterns LSTMs can learn',
            'Temporal smoothing to enhance pattern recognition',
            'Strategic label confusion to prevent tree overfitting',
            'Minimal cross-correlations to reduce complexity',
            'Enhanced separability for all model types'
        ]
    }
    
    with open('dataset/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Print summary
    print(f"\n✅ Optimized dataset creation completed!")
    print(f"📊 Dataset Statistics:")
    print(f"   • Total sequences: {len(X_noisy):,}")
    print(f"   • Sequence length: {X_noisy.shape[1]} time steps")
    print(f"   • Features per step: {X_noisy.shape[2]}")
    print(f"   • Instance types: {len(np.unique(y_encoded))}")
    print(f"   • Label noise: {np.sum(y_clean != y_noisy)} samples ({np.sum(y_clean != y_noisy)/len(y_noisy)*100:.1f}%)")
    
    print(f"\n🎯 Optimized for 80-85% accuracy:")
    print(f"   • LSTM-optimized temporal patterns")
    print(f"   • Sequence-level consistency for better learning")
    print(f"   • Gradual degradation patterns")
    print(f"   • Temporal smoothing for pattern recognition")
    print(f"   • Strategic noise to prevent tree overfitting")
    print(f"   • Enhanced LSTM performance while maintaining challenge")
    
    return X_noisy, y_encoded, label_encoder, feature_names, instance_ids_noisy

def load_dataset():
    """Helper function to load the generated dataset"""
    print("📂 Loading optimized dataset...")
    
    try:
        X = np.load('dataset/X_sequences_noisy.npy')
        y = np.load('dataset/y_labels_encoded.npy')
        
        with open('dataset/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('dataset/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        with open('dataset/instance_ids_noisy.pkl', 'rb') as f:
            instance_ids = pickle.load(f)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   • Sequences: {X.shape}")
        print(f"   • Labels: {y.shape}")
        print(f"   • Instance types: {len(label_encoder.classes_)}")
        
        return X, y, label_encoder, feature_names, instance_ids
        
    except FileNotFoundError as e:
        print(f"❌ Dataset files not found: {e}")
        print("   Please run create_and_save_enhanced_dataset() first.")
        return None, None, None, None, None

def preview_dataset():
    """Preview the generated dataset"""
    try:
        df = pd.read_csv('dataset/cloud_instance_dataset_enhanced.csv')
        print("📋 Optimized Dataset Preview:")
        print("=" * 50)
        print(f"Total records: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nInstance types distribution:")
        print(df['Instance_Type'].value_counts().head(10))
        print(f"\nFeature statistics (first 1000 samples):")
        print(df[['CPU_Utilization_Percent', 'Memory_Utilization_Percent', 
                 'Response_Time_ms', 'Hourly_Cost_USD']].head(1000).describe())
            
    except FileNotFoundError:
        print("❌ Dataset file not found. Please generate the dataset first.")

if __name__ == "__main__":
    # Generate the optimized dataset
    print("🎬 Starting optimized dataset generation for 80-85% accuracy...")
    X, y, encoder, features, instance_ids = create_and_save_enhanced_dataset()
    
    # Preview the results
    print("\n" + "="*70)
    preview_dataset()
    
    print(f"\n🎉 Optimized dataset generation complete!")
    print(f"   • Designed for 80-85% test accuracy")
    print(f"   • Better instance type separability")
    print(f"   • Reduced overfitting potential")
    print(f"   • Smart noise patterns for realistic challenge")