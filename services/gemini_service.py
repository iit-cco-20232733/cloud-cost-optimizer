"""
Gemini AI Service
Handles all interactions with Google Gemini API
"""
import json
from utils.config_loader import config

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False

class GeminiService:
    def __init__(self):
        self.gemini_config = config.get_gemini_config()
        self.api_key = self.gemini_config['api_key']
        self.model_name = self.gemini_config['model_name']
        self._client = None
        self._recommendation_cache = {}
    
    def get_client(self):
        """Get or create Gemini client"""
        if not GENAI_AVAILABLE:
            raise RuntimeError('google-genai package not installed.')
        if not self.api_key or self.api_key.startswith('REPLACE_WITH_'):
            raise RuntimeError('Gemini API key not configured.')
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    def get_instance_recommendation(self, metrics):
        """Get Gemini's instance type recommendation"""
        if not GENAI_AVAILABLE:
            return None
        
        cache_key = f"{metrics.get('CPU_Utilization_Percent', 0):.1f}_{metrics.get('Memory_Utilization_Percent', 0):.1f}"
        if cache_key in self._recommendation_cache:
            print(f"[CACHE] Using cached Gemini recommendation")
            return self._recommendation_cache[cache_key]
        
        try:
            client = self.get_client()
            metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
            
            prompt = f"""
You are an AWS EC2 instance sizing expert. Based on the following workload metrics, recommend the most appropriate AWS EC2 instance type.

Workload Metrics:
{metrics_text}

Return ONLY valid JSON with this exact schema:
{{
    "instance_type": "string",
    "reasoning": "string",
    "confidence": "high|medium|low"
}}

IMPORTANT:
1. Analyze the CPU and Memory utilization patterns
2. Consider network throughput requirements
3. Account for cost efficiency
4. Choose from common AWS instance types: t2.nano, t2.micro, t2.small, t2.medium, t3.nano, t3.micro, m5.large, m5.xlarge, m5.2xlarge, c5.large, c5.xlarge, c5.2xlarge, r5.large, r5.xlarge, x1.large
5. Provide clear reasoning for your recommendation
6. Be conservative - prefer slightly larger instances for stability
"""
            
            contents = [
                genai_types.Content(
                    role='user',
                    parts=[genai_types.Part.from_text(text=prompt)]
                )
            ]
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(response_mime_type='application/json')
            )
            
            raw_text = ''
            if hasattr(response, 'text') and response.text:
                raw_text = response.text
            elif getattr(response, 'candidates', None):
                for candidate in response.candidates:
                    candidate_parts = getattr(getattr(candidate, 'content', None), 'parts', [])
                    for part in candidate_parts:
                        raw_text += getattr(part, 'text', '')
            
            if raw_text:
                recommendation = json.loads(raw_text)
                if isinstance(recommendation, list):
                    recommendation = recommendation[0] if recommendation else {}
                
                result = {
                    'instance_type': recommendation.get('instance_type', 'unknown'),
                    'reasoning': recommendation.get('reasoning', 'No reasoning provided'),
                    'confidence': recommendation.get('confidence', 'medium')
                }
                
                self._recommendation_cache[cache_key] = result
                return result
            
            return None
            
        except Exception as e:
            print(f"[WARNING] Gemini recommendation failed: {e}")
            return None
    
    def get_batch_recommendations_and_costs(self, instances_data):
        """Get batch recommendations and costs for multiple instances"""
        if not GENAI_AVAILABLE or not instances_data:
            return {}
        
        try:
            client = self.get_client()
            pricing = config.get_pricing()
            
            instances_list = []
            for data in instances_data:
                instances_list.append({
                    'instance_id': data['instance_id'],
                    'current_type': data['current_type'],
                    'metrics': data['metrics']
                })
            
            # Create prompt with pricing reference
            pricing_examples = "\n  * ".join([f"{k}: ${v:.4f}/hr" for k, v in list(pricing.items())[:10]])
            
            prompt = f"""
You are an AWS EC2 optimization expert with access to current AWS pricing data. For each instance below:
1. Recommend the optimal AWS EC2 instance type based on workload metrics
2. Provide ACCURATE on-demand hourly costs for BOTH the current and recommended instance types
3. Explain your recommendation

CRITICAL PRICING REQUIREMENTS:
- Costs must be actual AWS us-east-1 on-demand pricing (as of 2024)
- Each instance type has a DIFFERENT cost - DO NOT return the same cost for different types
- Example pricing reference:
  * {pricing_examples}

Return ONLY valid JSON in this exact format:
{{
  "recommendations": [
    {{
      "instance_id": "string",
      "current_type": "string",
      "recommended_type": "string",
      "current_cost": 0.0,
      "recommended_cost": 0.0,
      "reasoning": "string",
      "confidence": "high|medium|low"
    }}
  ]
}}

RECOMMENDATION GUIDELINES:
- Analyze CPU%, Memory%, Network, Disk usage patterns
- If CPU < 30% AND Memory < 30%: Consider downsizing to save costs
- If CPU > 80% OR Memory > 80%: Recommend upgrade for performance
- If metrics are balanced: Keep current or suggest cost-effective alternative
- Choose from: t2.nano, t2.micro, t2.small, t2.medium, t3.nano, t3.micro, t3.small, t3.medium, m5.large, m5.xlarge, m5.2xlarge, c5.large, c5.xlarge, c5.2xlarge, r5.large, r5.xlarge, r5.2xlarge, x1.large

Instances and their workload metrics:
{json.dumps(instances_list, indent=2)}
"""
            
            contents = [
                genai_types.Content(
                    role='user',
                    parts=[genai_types.Part.from_text(text=prompt)]
                )
            ]
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(response_mime_type='application/json')
            )
            
            raw_text = ''
            if hasattr(response, 'text') and response.text:
                raw_text = response.text
            elif getattr(response, 'candidates', None):
                for candidate in response.candidates:
                    candidate_parts = getattr(getattr(candidate, 'content', None), 'parts', [])
                    for part in candidate_parts:
                        raw_text += getattr(part, 'text', '')
            
            if raw_text:
                batch_result = json.loads(raw_text)
                if isinstance(batch_result, list):
                    batch_result = batch_result[0] if batch_result else {}
                
                recommendations = {}
                for rec in batch_result.get('recommendations', []):
                    instance_id = rec.get('instance_id')
                    current_type = rec.get('current_type', 'unknown')
                    recommended_type = rec.get('recommended_type', 'unknown')
                    
                    if instance_id:
                        # Validate costs
                        current_cost_raw = float(rec.get('current_cost', 0))
                        recommended_cost_raw = float(rec.get('recommended_cost', 0))
                        
                        current_cost = self._validate_cost(current_type, current_cost_raw)
                        recommended_cost = self._validate_cost(recommended_type, recommended_cost_raw)
                        
                        recommendations[instance_id] = {
                            'recommended_type': recommended_type,
                            'current_cost': current_cost,
                            'recommended_cost': recommended_cost,
                            'reasoning': rec.get('reasoning', 'No reasoning provided'),
                            'confidence': rec.get('confidence', 'medium')
                        }
                        print(f"[GEMINI] {instance_id}: {current_type} → {recommended_type} (${current_cost:.4f} → ${recommended_cost:.4f})")
                
                return recommendations
            
            return {}
            
        except Exception as e:
            print(f"[WARNING] Gemini batch failed: {e}")
            return {}
    
    def _validate_cost(self, instance_type, gemini_cost):
        """Validate Gemini cost against reference pricing"""
        reference_cost = config.get_instance_price(instance_type)
        
        if gemini_cost == 0:
            return reference_cost
        
        if reference_cost > 0:
            diff_percent = abs(gemini_cost - reference_cost) / reference_cost * 100
            if diff_percent > 50:
                print(f"[WARNING] Gemini cost ${gemini_cost:.4f} for {instance_type} differs {diff_percent:.1f}% from reference ${reference_cost:.4f}, using reference")
                return reference_cost
        
        return gemini_cost

    def get_instance_comparison(self, instance_id, current_type, predicted_type, metrics):
        """Get detailed instance comparison with specs from Gemini"""
        if not GENAI_AVAILABLE:
            return self._build_fallback(current_type, predicted_type, metrics)
        
        try:
            client = self.get_client()
            
            # Build prompt
            metrics_payload = {
                metric: {
                    'current': value,
                    'predicted': value,
                    'difference': 0,
                    'difference_percent': 0
                }
                for metric, value in metrics.items()
            }
            
            prompt = f"""
You are an AWS EC2 instance optimization expert. Analyze the current instance and provide a detailed comparison with the recommended instance type.

Return ONLY valid JSON with this schema:
{{
    "caption": "string",
    "disclaimer": "string",
    "rows": [
        {{
            "metric": "string",
            "current": "string",
            "recommended": "string",
            "unit": "string"
        }}
    ],
    "recommendations": ["string", ...]
}}

Mandatory rows to include:
1. "vCPU" - Number of virtual CPUs
2. "Instance Memory" - RAM in GiB
3. "Network Performance" - Network bandwidth
4. "Hourly Cost (On demand)" - AWS pricing in USD/hr
5. "Category" - Instance category
6. "CPU Utilization" - Current usage from telemetry
7. "Memory Utilization" - Current usage from telemetry

Instance: {instance_id}
Current type: {current_type}
Recommended type: {predicted_type}

Workload metrics:
{json.dumps(metrics_payload, indent=2)}

Provide accurate AWS EC2 specs and actionable recommendations.
"""
            
            contents = [
                genai_types.Content(
                    role='user',
                    parts=[genai_types.Part.from_text(text=prompt)]
                )
            ]
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(response_mime_type='application/json')
            )
            
            raw_text = ''
            if hasattr(response, 'text') and response.text:
                raw_text = response.text
            elif getattr(response, 'candidates', None):
                for candidate in response.candidates:
                    candidate_parts = getattr(getattr(candidate, 'content', None), 'parts', [])
                    for part in candidate_parts:
                        raw_text += getattr(part, 'text', '')
            
            if raw_text:
                result = json.loads(raw_text)
                result.setdefault('caption', 'Gemini-assisted comparison')
                result.setdefault('disclaimer', 'Generated with Gemini')
                result.setdefault('generated', True)
                return result
            
            return self._build_fallback(current_type, predicted_type, metrics)
            
        except Exception as e:
            print(f"[WARNING] Gemini comparison failed: {e}")
            return self._build_fallback(current_type, predicted_type, metrics)
    
    def _build_fallback(self, current_type, predicted_type, metrics):
        """Build fallback response when Gemini is unavailable"""
        cpu_util = metrics.get('CPU_Utilization_Percent', 0)
        mem_util = metrics.get('Memory_Utilization_Percent', 0)
        
        return {
            'rows': [
                {
                    'metric': 'Current Instance',
                    'current': current_type,
                    'recommended': predicted_type,
                    'unit': 'Type'
                },
                {
                    'metric': 'CPU Utilization',
                    'current': f"{cpu_util:.1f}%",
                    'recommended': '—',
                    'unit': '%'
                },
                {
                    'metric': 'Memory Utilization',
                    'current': f"{mem_util:.1f}%",
                    'recommended': '—',
                    'unit': '%'
                }
            ],
            'caption': 'Basic comparison (Gemini unavailable)',
            'disclaimer': 'Gemini AI is unavailable. Showing basic metrics only.',
            'recommendations': [
                'Gemini AI provides detailed instance specifications.',
                'Check your API configuration or try again later.'
            ],
            'generated': False
        }

# Singleton instance
gemini_service = GeminiService()

