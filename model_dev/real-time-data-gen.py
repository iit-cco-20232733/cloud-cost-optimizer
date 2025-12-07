"""
Monthly Cloud Instance Dataset Generator (Daily granularity, lifecycle-aware)
- Enforces 10–20 instances per month (any types).
- Each instance has a start_day..end_day lifecycle within the month (some full month, some partial, some short).
- Emits ONE row per active (instance, day). Keeps original column names and metric semantics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import calendar
import os
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class MonthlyCloudInstanceGenerator:
    def __init__(self):
        """Initialize the monthly cloud instance dataset generator with the same metrics and names."""
        # ==== Instance type catalog (unchanged) ====
        self.instance_types = {
            # T-series (Burstable)
            't2.nano': {'cpu_base': 12, 'cpu_std': 6, 'memory_base': 15, 'memory_std': 8,
                        'network_in_base': 8, 'network_in_std': 3, 'network_out_base': 8, 'network_out_std': 3,
                        'network_throughput_base': 15, 'network_throughput_std': 5,
                        'disk_io_base': 120, 'disk_io_std': 30, 'disk_usage_base': 20, 'disk_usage_std': 10,
                        'response_time_base': 280, 'response_time_std': 40, 'cost_base': 0.0058, 'cost_std': 0.001},
            't2.micro': {'cpu_base': 22, 'cpu_std': 8, 'memory_base': 25, 'memory_std': 10,
                         'network_in_base': 12, 'network_in_std': 4, 'network_out_base': 12, 'network_out_std': 4,
                         'network_throughput_base': 22, 'network_throughput_std': 6,
                         'disk_io_base': 200, 'disk_io_std': 40, 'disk_usage_base': 30, 'disk_usage_std': 12,
                         'response_time_base': 220, 'response_time_std': 35, 'cost_base': 0.0116, 'cost_std': 0.002},
            't2.small': {'cpu_base': 32, 'cpu_std': 10, 'memory_base': 35, 'memory_std': 12,
                         'network_in_base': 18, 'network_in_std': 5, 'network_out_base': 18, 'network_out_std': 5,
                         'network_throughput_base': 32, 'network_throughput_std': 8,
                         'disk_io_base': 320, 'disk_io_std': 60, 'disk_usage_base': 40, 'disk_usage_std': 15,
                         'response_time_base': 180, 'response_time_std': 30, 'cost_base': 0.023, 'cost_std': 0.003},
            't2.medium': {'cpu_base': 45, 'cpu_std': 12, 'memory_base': 50, 'memory_std': 15,
                          'network_in_base': 28, 'network_in_std': 7, 'network_out_base': 28, 'network_out_std': 7,
                          'network_throughput_base': 50, 'network_throughput_std': 12,
                          'disk_io_base': 500, 'disk_io_std': 80, 'disk_usage_base': 55, 'disk_usage_std': 18,
                          'response_time_base': 140, 'response_time_std': 25, 'cost_base': 0.046, 'cost_std': 0.004},
            't3.nano': {'cpu_base': 15, 'cpu_std': 6, 'memory_base': 18, 'memory_std': 8,
                        'network_in_base': 10, 'network_in_std': 3, 'network_out_base': 10, 'network_out_std': 3,
                        'network_throughput_base': 18, 'network_throughput_std': 5,
                        'disk_io_base': 150, 'disk_io_std': 35, 'disk_usage_base': 22, 'disk_usage_std': 10,
                        'response_time_base': 250, 'response_time_std': 35, 'cost_base': 0.0052, 'cost_std': 0.001},
            't3.micro': {'cpu_base': 25, 'cpu_std': 8, 'memory_base': 28, 'memory_std': 10,
                         'network_in_base': 14, 'network_in_std': 4, 'network_out_base': 14, 'network_out_std': 4,
                         'network_throughput_base': 25, 'network_throughput_std': 6,
                         'disk_io_base': 220, 'disk_io_std': 45, 'disk_usage_base': 32, 'disk_usage_std': 12,
                         'response_time_base': 200, 'response_time_std': 30, 'cost_base': 0.0104, 'cost_std': 0.002},
            # M-series
            'm5.large': {'cpu_base': 65, 'cpu_std': 15, 'memory_base': 70, 'memory_std': 18,
                         'network_in_base': 55, 'network_in_std': 12, 'network_out_base': 55, 'network_out_std': 12,
                         'network_throughput_base': 110, 'network_throughput_std': 20,
                         'disk_io_base': 1100, 'disk_io_std': 150, 'disk_usage_base': 72, 'disk_usage_std': 20,
                         'response_time_base': 85, 'response_time_std': 18, 'cost_base': 0.096, 'cost_std': 0.008},
            'm5.xlarge': {'cpu_base': 78, 'cpu_std': 15, 'memory_base': 82, 'memory_std': 18,
                          'network_in_base': 70, 'network_in_std': 15, 'network_out_base': 70, 'network_out_std': 15,
                          'network_throughput_base': 155, 'network_throughput_std': 25,
                          'disk_io_base': 1550, 'disk_io_std': 200, 'disk_usage_base': 80, 'disk_usage_std': 22,
                          'response_time_base': 65, 'response_time_std': 15, 'cost_base': 0.192, 'cost_std': 0.015},
            'm5.2xlarge': {'cpu_base': 88, 'cpu_std': 12, 'memory_base': 90, 'memory_std': 15,
                           'network_in_base': 90, 'network_in_std': 18, 'network_out_base': 90, 'network_out_std': 18,
                           'network_throughput_base': 220, 'network_throughput_std': 30,
                           'disk_io_base': 2200, 'disk_io_std': 250, 'disk_usage_base': 85, 'disk_usage_std': 20,
                           'response_time_base': 48, 'response_time_std': 12, 'cost_base': 0.384, 'cost_std': 0.020},
            # C-series
            'c5.large': {'cpu_base': 82, 'cpu_std': 12, 'memory_base': 58, 'memory_std': 15,
                         'network_in_base': 68, 'network_in_std': 15, 'network_out_base': 68, 'network_out_std': 15,
                         'network_throughput_base': 135, 'network_throughput_std': 22,
                         'disk_io_base': 1350, 'disk_io_std': 180, 'disk_usage_base': 62, 'disk_usage_std': 18,
                         'response_time_base': 55, 'response_time_std': 12, 'cost_base': 0.085, 'cost_std': 0.008},
            'c5.xlarge': {'cpu_base': 90, 'cpu_std': 10, 'memory_base': 68, 'memory_std': 15,
                          'network_in_base': 85, 'network_in_std': 18, 'network_out_base': 85, 'network_out_std': 18,
                          'network_throughput_base': 180, 'network_throughput_std': 25,
                          'disk_io_base': 1800, 'disk_io_std': 220, 'disk_usage_base': 70, 'disk_usage_std': 18,
                          'response_time_base': 42, 'response_time_std': 10, 'cost_base': 0.17, 'cost_std': 0.012},
            'c5.2xlarge': {'cpu_base': 94, 'cpu_std': 8, 'memory_base': 75, 'memory_std': 15,
                           'network_in_base': 105, 'network_in_std': 20, 'network_out_base': 105, 'network_out_std': 20,
                           'network_throughput_base': 240, 'network_throughput_std': 30,
                           'disk_io_base': 2400, 'disk_io_std': 280, 'disk_usage_base': 75, 'disk_usage_std': 18,
                           'response_time_base': 32, 'response_time_std': 8, 'cost_base': 0.34, 'cost_std': 0.018},
            # R-series
            'r5.large': {'cpu_base': 62, 'cpu_std': 15, 'memory_base': 88, 'memory_std': 15,
                         'network_in_base': 52, 'network_in_std': 12, 'network_out_base': 52, 'network_out_std': 12,
                         'network_throughput_base': 105, 'network_throughput_std': 20,
                         'disk_io_base': 1050, 'disk_io_std': 140, 'disk_usage_base': 85, 'disk_usage_std': 18,
                         'response_time_base': 75, 'response_time_std': 15, 'cost_base': 0.126, 'cost_std': 0.010},
            'r5.xlarge': {'cpu_base': 75, 'cpu_std': 15, 'memory_base': 92, 'memory_std': 12,
                          'network_in_base': 72, 'network_in_std': 15, 'network_out_base': 72, 'network_out_std': 15,
                          'network_throughput_base': 145, 'network_throughput_std': 22,
                          'disk_io_base': 1450, 'disk_io_std': 180, 'disk_usage_base': 88, 'disk_usage_std': 15,
                          'response_time_base': 62, 'response_time_std': 12, 'cost_base': 0.252, 'cost_std': 0.015},
            # X-series
            'x1.large': {'cpu_base': 80, 'cpu_std': 12, 'memory_base': 95, 'memory_std': 10,
                         'network_in_base': 65, 'network_in_std': 12, 'network_out_base': 65, 'network_out_std': 12,
                         'network_throughput_base': 130, 'network_throughput_std': 20,
                         'disk_io_base': 1300, 'disk_io_std': 160, 'disk_usage_base': 92, 'disk_usage_std': 12,
                         'response_time_base': 50, 'response_time_std': 10, 'cost_base': 0.334, 'cost_std': 0.020}
        }

        self.months = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December']

        self.seasonal_factors = {
            'January': 1.1, 'February': 1.0, 'March': 1.05, 'April': 1.15, 'May': 1.2, 'June': 1.25,
            'July': 1.3, 'August': 1.28, 'September': 1.22, 'October': 1.18, 'November': 1.12, 'December': 1.08
        }

        # Prebuild a global pool of instance IDs per type (20–50 per type), reusable across months
        self.instance_pools = {t: self._make_instance_ids() for t in self.instance_types.keys()}

    def _make_instance_ids(self, n_min=20, n_max=50):
        hex_chars = '0123456789abcdef'
        ids = []
        for _ in range(random.randint(n_min, n_max)):
            ids.append('i-' + ''.join(random.choices(hex_chars, k=17)))
        return ids

    def _days_in_month(self, month, year=2024):
        m = self.months.index(month) + 1
        return calendar.monthrange(year, m)[1]

    # ---------- NEW: instance selection & lifecycle logic ----------
    def _select_instances_for_month(self, month):
        """
        Select 10–20 unique (instance_id, instance_type) pairs across ALL types for the given month.
        """
        target_n = random.randint(10, 20)
        selected = set()
        types_list = list(self.instance_types.keys())

        while len(selected) < target_n:
            t = random.choice(types_list)
            iid = random.choice(self.instance_pools[t])
            selected.add((iid, t))  # uniqueness within month
        return list(selected)

    def _choose_lifecycle(self, last_day):
        """
        Choose a lifecycle window (start_day, end_day) inside the month.
        - ~30% full month
        - ~45% partial (7–25 days)
        - ~25% very short (1–6 days)
        """
        r = random.random()
        if r < 0.30:
            return 1, last_day  # full month
        elif r < 0.75:
            length = random.randint(7, min(25, last_day))
        else:
            length = random.randint(1, min(6, last_day))
        start = random.randint(1, last_day - length + 1)
        end = start + length - 1
        return start, end

    # ---------- metric synthesis (daily) ----------
    def _emit_daily_record(self, year, month_name, day, instance_id, instance_type, config, seasonal_factor, last_day):
        """
        Create ONE daily record using the same metric names and semantics as the original code.
        Hour fixed at 0; Business_Hours=False for daily rows.
        """
        # Weekend factor
        dt = datetime(year, self.months.index(month_name) + 1, day)
        weekend = dt.weekday() >= 5  # Saturday/Sunday
        weekend_factor = 0.7 if weekend else 1.0

        # Daily variation across the month (smooth sinusoidal)
        daily_factor = 1 + 0.15 * np.sin(2 * np.pi * day / last_day)

        # Small random daily jitter to avoid uniformity
        jitter = np.random.normal(1.0, 0.03)

        total_factor = seasonal_factor * daily_factor * weekend_factor * jitter

        # Metrics (same names, clipped to sane bounds)
        cpu_util = np.clip(np.random.normal(config['cpu_base'] * total_factor, config['cpu_std']), 0, 100)
        memory_util = np.clip(np.random.normal(config['memory_base'] * total_factor, config['memory_std']), 0, 100)
        network_in = np.clip(np.random.normal(config['network_in_base'] * total_factor, config['network_in_std']), 0, 10000)
        network_out = np.clip(np.random.normal(config['network_out_base'] * total_factor, config['network_out_std']), 0, 10000)
        net_tput = np.clip(np.random.normal(config['network_throughput_base'] * total_factor, config['network_throughput_std']), 0, 20000)
        disk_io = np.clip(np.random.normal(config['disk_io_base'] * total_factor, config['disk_io_std']), 0, 20000)

        # Disk usage grows across the **month** (keep original intent)
        disk_growth = (day / last_day) * 10.0  # up to +10% by month end
        disk_usage = np.clip(np.random.normal(config['disk_usage_base'] + disk_growth, config['disk_usage_std']), 0, 100)

        # Response time inversely related to load factor in original code; keep same shape
        resp_time = np.clip(np.random.normal(config['response_time_base'] / max(total_factor, 0.1), config['response_time_std']), 10, 2000)

        hourly_cost = np.clip(np.random.normal(config['cost_base'] * total_factor, config['cost_std']), 0.001, 5.0)

        timestamp = f"2024-{self.months.index(month_name)+1:02d}-{day:02d} 00:00:00"

        return {
            'Timestamp': timestamp,
            'Month': month_name,
            'Day': day,
            'Hour': 0,  # fixed for daily granularity
            'Instance_ID': instance_id,
            'Instance_Type': instance_type,
            'CPU_Utilization_Percent': round(cpu_util, 2),
            'Memory_Utilization_Percent': round(memory_util, 2),
            'Network_In_Mbps': round(network_in, 2),
            'Network_Out_Mbps': round(network_out, 2),
            'Network_Throughput_Mbps': round(net_tput, 2),
            'Disk_IO_IOPS': round(disk_io, 2),
            'Disk_Usage_Percent': round(disk_usage, 2),
            'Response_Time_ms': round(resp_time, 2),
            'Hourly_Cost_USD': round(hourly_cost, 4),
            'Seasonal_Factor': round(seasonal_factor, 2),
            'Weekend': weekend,
            'Business_Hours': False  # daily rows—not applicable
        }

    # ---------- public generators ----------
    def generate_monthly_data(self, month):
        """
        Generate ONE month:
        - pick 10–20 instances (any type)
        - assign lifecycle windows
        - emit 1 row/day for each instance in its active window
        """
        print(f"Generating data for {month}...")
        last_day = self._days_in_month(month, 2024)
        seasonal_factor = self.seasonal_factors[month]
        year = 2024

        # Select instances
        pairs = self._select_instances_for_month(month)

        # For each instance, choose lifecycle and emit daily rows inside [start,end]
        rows = []
        lifecycles = {}  # for summary
        for instance_id, instance_type in pairs:
            start_day, end_day = self._choose_lifecycle(last_day)
            lifecycles[(instance_id, instance_type)] = (start_day, end_day)
            config = self.instance_types[instance_type]

            for day in range(start_day, end_day + 1):
                rows.append(
                    self._emit_daily_record(year, month, day, instance_id, instance_type, config, seasonal_factor, last_day)
                )

        # Summary
        print(f"  • Instances selected: {len(pairs)} (target 10–20)")
        full = sum(1 for (_, _), (s, e) in lifecycles.items() if s == 1 and e == last_day)
        print(f"  • Lifecycles: full-month={full}, partial={len(pairs)-full}")
        print(f"  • Rows generated: {len(rows):,}")

        return rows, pairs, lifecycles

    def generate_full_year_dataset(self):
        print("🚀 Starting Full Year Cloud Instance Dataset Generation (DAILY, lifecycle-aware)")
        print("📋 Each month will use 10–20 randomly selected instances (any types)")
        print("=" * 70)

        all_rows = []
        monthly_selected = {}
        monthly_lifecycles = {}

        for month in self.months:
            rows, pairs, lifecycles = self.generate_monthly_data(month)
            all_rows.extend(rows)
            monthly_selected[month] = pairs
            monthly_lifecycles[month] = lifecycles

        df = pd.DataFrame(all_rows).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n✅ Full year dataset generated!")
        print(f"   • Total records: {len(df):,}")
        print(f"   • Unique instances (year): {df['Instance_ID'].nunique()}")
        print(f"   • Time range: 2024-01-01 .. 2024-12-31 (daily)")
        print("\n📊 Monthly Instance Counts (should be 10–20):")
        for m in self.months:
            print(f"   • {m}: {len(set(monthly_selected[m]))} instances")

        return df, monthly_selected, monthly_lifecycles

    def save_dataset(self, df, filename='monthly_cloud_instances_2024.csv'):
        os.makedirs('monthly_dataset', exist_ok=True)
        path = os.path.join('monthly_dataset', filename)
        df.to_csv(path, index=False)
        print(f"\n💾 Dataset saved to: {path}")
        return path

    def validate_dataset(self, df, monthly_selected, monthly_lifecycles):
        print("\n🔍 Dataset Validation")
        print("=" * 50)
        # Basic checks
        print(f"✅ Total records: {len(df):,}")
        print(f"✅ Missing values: {df.isnull().sum().sum()}")
        print(f"✅ Duplicate rows: {df.duplicated().sum()}")

        # Check per-month instance counts (hard constraint)
        ok = True
        print("\n📅 Per-month instance count checks (must be 10–20):")
        for m in self.months:
            n = len(set(monthly_selected[m]))
            cond = 10 <= n <= 20
            print(f"   • {m}: {n} -> {'OK' if cond else 'VIOLATION'}")
            ok = ok and cond
        if not ok:
            print("❌ Constraint violation: some months are outside 10–20 instances.")

        # Lifecycle boundaries respected
        print("\n🧭 Lifecycle boundary spot-check:")
        for m in self.months[:3]:  # sample a few
            lifecycles = monthly_lifecycles[m]
            last_day = self._days_in_month(m, 2024)
            bad = [k for k, (s, e) in lifecycles.items() if not (1 <= s <= e <= last_day)]
            print(f"   • {m}: {len(bad)} invalid lifecycles")

        # Feature ranges (same metrics)
        print("\n📊 Feature ranges (key columns):")
        for col in ['CPU_Utilization_Percent','Memory_Utilization_Percent',
                    'Network_In_Mbps','Network_Out_Mbps',
                    'Network_Throughput_Mbps','Disk_IO_IOPS',
                    'Disk_Usage_Percent','Response_Time_ms','Hourly_Cost_USD']:
            print(f"   • {col}: min={df[col].min():.2f} max={df[col].max():.2f} mean={df[col].mean():.2f}")

def main():
    print("🌟 Monthly Cloud Instance Dataset Generator (Daily, lifecycle-aware)")
    print("=" * 70)
    gen = MonthlyCloudInstanceGenerator()

    # Build full-year dataset (daily rows)
    df, monthly_selected, monthly_lifecycles = gen.generate_full_year_dataset()

    # Validate
    gen.validate_dataset(df, monthly_selected, monthly_lifecycles)

    # Save
    path = gen.save_dataset(df)

    # Quick analytics (unchanged style)
    print(f"\n📈 Quick Analytics:")
    print(f"   • Avg CPU utilization: {df['CPU_Utilization_Percent'].mean():.2f}%")
    print(f"   • Avg memory utilization: {df['Memory_Utilization_Percent'].mean():.2f}%")
    print(f"   • Avg response time: {df['Response_Time_ms'].mean():.2f} ms")
    print(f"   • Avg hourly cost: ${df['Hourly_Cost_USD'].mean():.4f}")

    print(f"\n📋 Sample:")
    print(df[['Timestamp','Instance_ID','Instance_Type',
              'CPU_Utilization_Percent','Memory_Utilization_Percent','Response_Time_ms']].head(10))

    print("\n🎉 Generation complete. Files in 'monthly_dataset/'")
    return df

if __name__ == "__main__":
    dataset = main()
