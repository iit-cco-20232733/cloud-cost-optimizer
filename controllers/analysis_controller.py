"""
Analysis Controller
Handles instance analysis and provisioning logic
"""
import pandas as pd
from services import gemini_service, lstm_service
from utils.config_loader import config

class AnalysisController:
    def __init__(self):
        self.model_features = config.get_model_features()
    
    def analyze_provisioning(self, current_type, lstm_type, gemini_type, current_metrics, 
                           cost_current=None, cost_lstm=None, cost_gemini=None):
        """Analyze provisioning status by comparing LSTM and Gemini recommendations"""
        
        cpu_util = current_metrics.get('CPU_Utilization_Percent', 0)
        memory_util = current_metrics.get('Memory_Utilization_Percent', 0)
        
        # Validate and fix costs using reference pricing
        if cost_current is None or cost_current == 0:
            cost_current = config.get_instance_price(current_type)
            if cost_current == 0:
                print(f"[WARNING] No pricing data for {current_type}")
        
        if cost_lstm is None or cost_lstm == 0:
            cost_lstm = config.get_instance_price(lstm_type)
            if cost_lstm == 0:
                print(f"[WARNING] No pricing data for {lstm_type}")
        
        if cost_gemini is None or cost_gemini == 0:
            cost_gemini = config.get_instance_price(gemini_type)
            if cost_gemini == 0:
                print(f"[WARNING] No pricing data for {gemini_type}")
        
        print(f"[COST] Current: {current_type} ${cost_current:.4f}/hr | LSTM: {lstm_type} ${cost_lstm:.4f}/hr | Gemini: {gemini_type} ${cost_gemini:.4f}/hr")
        
        # Determine recommendation (prefer Gemini if available)
        models_agree = lstm_type == gemini_type
        recommended_type = gemini_type
        recommended_cost = cost_gemini
        
        # Calculate monthly savings (730 hours per month)
        # Positive = current costs MORE than recommended (can save by switching)
        # Negative = current costs LESS than recommended (would cost more to switch)
        monthly_savings = (cost_current - recommended_cost) * 730
        
        # Determine status based on cost comparison and utilization
        if current_type == recommended_type:
            status = 'optimal'
            recommendation = 'Current instance type is optimal'
            monthly_savings = 0
        else:
            # Status is determined by COST relationship, not utilization
            # Over-provisioned = Current is MORE expensive than needed (can save money)
            # Under-provisioned = Current is LESS expensive (need to spend more for better performance)
            
            if monthly_savings > 5:  # Current costs MORE → Over-provisioned
                status = 'over-provisioned'
                if cpu_util < 40 and memory_util < 40:
                    recommendation = f'Low utilization ({cpu_util:.1f}% CPU, {memory_util:.1f}% RAM). Downsize to {recommended_type} to save ${monthly_savings:.2f}/month'
                else:
                    recommendation = f'Can optimize to {recommended_type} and save ${monthly_savings:.2f}/month'
            
            elif monthly_savings < -5:  # Current costs LESS → Under-provisioned
                status = 'under-provisioned'
                if cpu_util > 75 or memory_util > 75:
                    recommendation = f'High utilization ({cpu_util:.1f}% CPU, {memory_util:.1f}% RAM). Upgrade to {recommended_type} (${abs(monthly_savings):.2f}/month additional)'
                else:
                    recommendation = f'Upgrade to {recommended_type} recommended (${abs(monthly_savings):.2f}/month additional)'
            
            else:  # Marginal difference (-$5 to +$5)
                status = 'optimal'
                if cpu_util > 75 or memory_util > 75:
                    recommendation = f'High utilization but minimal cost difference. Monitor performance.'
                elif cpu_util < 40 and memory_util < 40:
                    recommendation = f'Low utilization but minimal cost savings. Current instance acceptable.'
                else:
                    recommendation = f'Current instance adequate (minimal cost difference)'
        
        metrics_comparison = self._compare_metrics(current_metrics)
        
        return {
            'status': status,
            'recommendation': recommendation,
            'current_type': current_type,
            'lstm_type': lstm_type,
            'gemini_type': gemini_type,
            'models_agree': models_agree,
            'current_cost': cost_current,
            'lstm_cost': cost_lstm,
            'gemini_cost': cost_gemini,
            'potential_savings_lstm': round(cost_current - cost_lstm, 4),
            'potential_savings_gemini': round(cost_current - cost_gemini, 4),
            'potential_monthly_savings': round(monthly_savings, 2),
            'metrics_comparison': metrics_comparison
        }
    
    def _compare_metrics(self, current_metrics, predicted_metrics=None):
        """Compare current metrics with predicted metrics"""
        if predicted_metrics is None:
            predicted_metrics = {}
        
        comparison = {}
        for metric in self.model_features:
            current_value = current_metrics.get(metric, 0)
            predicted_value = predicted_metrics.get(metric, 0)
            
            if predicted_value > 0:
                difference_percent = ((current_value - predicted_value) / predicted_value) * 100
            else:
                difference_percent = 0
            
            comparison[metric] = {
                'current': round(current_value, 2),
                'predicted': round(predicted_value, 2),
                'difference': round(current_value - predicted_value, 2),
                'difference_percent': round(difference_percent, 2),
                'status': 'higher' if current_value > predicted_value else 'lower' if current_value < predicted_value else 'equal'
            }
        
        return comparison
    
    def analyze_month(self, month_data):
        """Analyze all instances for a month"""
        if month_data is None or len(month_data) == 0:
            return None
        
        # Calculate instance averages
        instance_averages = self._calculate_instance_averages(month_data)
        
        if len(instance_averages) == 0:
            return None
        
        results = []
        status_counts = {'optimal': 0, 'over-provisioned': 0, 'under-provisioned': 0}
        
        # Step 1: Collect LSTM predictions
        instances_for_gemini = []
        lstm_predictions_map = {}
        
        for _, row in instance_averages.iterrows():
            instance_id = row['Instance_ID']
            current_type = row['Instance_Type']
            
            # Extract metrics
            metrics = {feature: row.get(feature, 0) for feature in self.model_features}
            
            # Get LSTM prediction
            lstm_prediction = lstm_service.predict(metrics)
            if not lstm_prediction:
                continue
            
            lstm_predictions_map[instance_id] = lstm_prediction
            
            instances_for_gemini.append({
                'instance_id': instance_id,
                'current_type': current_type,
                'metrics': metrics
            })
        
        # Step 2: Batch Gemini request
        print(f"[BATCH] Fetching recommendations for {len(instances_for_gemini)} instances")
        gemini_batch_results = gemini_service.get_batch_recommendations_and_costs(instances_for_gemini)
        
        # Step 3: Process results
        for data in instances_for_gemini:
            instance_id = data['instance_id']
            current_type = data['current_type']
            metrics = data['metrics']
            
            lstm_prediction = lstm_predictions_map.get(instance_id)
            if not lstm_prediction:
                continue
            
            lstm_type = lstm_prediction['instance_type']
            lstm_confidence = lstm_prediction['confidence']
            
            # Get Gemini results
            gemini_result = gemini_batch_results.get(instance_id)
            if gemini_result:
                gemini_type = gemini_result['recommended_type']
                gemini_reasoning = gemini_result['reasoning']
                gemini_confidence = gemini_result['confidence']
                cost_current = gemini_result.get('current_cost', 0)
                cost_gemini = gemini_result.get('recommended_cost', 0)
                
                # Get LSTM cost
                if lstm_type == gemini_type:
                    cost_lstm = cost_gemini
                elif lstm_type == current_type:
                    cost_lstm = cost_current
                else:
                    cost_lstm = config.get_instance_price(lstm_type)
            else:
                gemini_type = lstm_type
                gemini_reasoning = "Gemini unavailable"
                gemini_confidence = "unknown"
                cost_current = 0
                cost_lstm = 0
                cost_gemini = 0
            
            # Analyze
            analysis = self.analyze_provisioning(
                current_type, lstm_type, gemini_type, metrics,
                cost_current, cost_lstm, cost_gemini
            )
            
            status_counts[analysis['status']] += 1
            
            results.append({
                'instance_id': instance_id,
                'current_type': current_type,
                'lstm_prediction': {
                    'instance_type': lstm_type,
                    'confidence': round(lstm_confidence * 100, 2),
                    'probabilities': lstm_prediction.get('probabilities', [])
                },
                'gemini_prediction': {
                    'instance_type': gemini_type,
                    'reasoning': gemini_reasoning,
                    'confidence': gemini_confidence
                },
                'models_agree': lstm_type == gemini_type,
                'metrics': {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                'analysis': analysis
            })
        
        return {
            'results': results,
            'status_counts': status_counts,
            'total_instances': len(results)
        }
    
    def _calculate_instance_averages(self, month_data):
        """Calculate average metrics per instance"""
        grouped = month_data.groupby(['Instance_ID', 'Instance_Type']).agg({
            feature: 'mean' for feature in self.model_features
        }).reset_index()
        
        return grouped

# Singleton instance
analysis_controller = AnalysisController()
