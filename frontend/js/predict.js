const MODEL_FEATURES = [
  'Network_In_Mbps',
  'Network_Out_Mbps',
  'Response_Time_ms',
  'CPU_Utilization_Percent',
  'Memory_Utilization_Percent',
  'Disk_Usage_Percent'
];

let standardMetrics = {};

async function loadStandardMetrics() {
  try {
    const response = await fetch('/api/instance_standard_metrics');
    if (!response.ok) {
      throw new Error('Unable to load standard metrics');
    }
    standardMetrics = await response.json();
  } catch (error) {
    console.error(error);
  }
}

function renderPrediction(result) {
  const title = document.getElementById('prediction-title');
  const subtitle = document.getElementById('prediction-subtitle');
  const metadata = document.getElementById('prediction-metadata');

  const prediction = result?.prediction;
  if (!prediction) {
    title.textContent = 'Prediction failed';
    subtitle.textContent = result?.error ?? 'The backend could not compute a response.';
    metadata.innerHTML = '';
    return;
  }

  title.textContent = prediction.instance_type;
  subtitle.textContent = `Predicted instance type based on workload metrics`;

  const info = result.instance_info ?? {};
  metadata.innerHTML = `
   
  `;
}

function renderComparison(result) {
  const card = document.getElementById('comparison-card');
  card.classList.remove('hidden');
  
  // Display LSTM prediction
  const lstmInstance = document.getElementById('lstm-instance');
  lstmInstance.textContent = result?.prediction?.instance_type || '—';
  
  // Display Gemini recommendation
  const geminiInstance = document.getElementById('gemini-instance');
  const geminiConfidence = document.getElementById('gemini-confidence');
  const geminiReasoningContainer = document.getElementById('gemini-reasoning-container');
  const geminiReasoning = document.getElementById('gemini-reasoning');
  
  if (result?.gemini_recommendation) {
    const gemini = result.gemini_recommendation;
    geminiInstance.textContent = gemini.instance_type || '—';
    
    // Show confidence
    if (gemini.confidence) {
      const confidenceText = gemini.confidence.charAt(0).toUpperCase() + gemini.confidence.slice(1);
      geminiConfidence.textContent = `Confidence: ${confidenceText}`;
    }
    
    // Show reasoning
    if (gemini.reasoning) {
      geminiReasoning.textContent = gemini.reasoning;
      geminiReasoningContainer.style.display = 'block';
    }
  } else {
    geminiInstance.textContent = 'Not available';
    geminiConfidence.textContent = '';
    geminiReasoningContainer.style.display = 'none';
  }
}

async function submitPrediction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const error = document.getElementById('form-error');
  error.classList.add('hidden');

  const payload = {};
  MODEL_FEATURES.forEach((field) => {
    const value = form.elements[field]?.value;
    payload[field] = Number(value ?? 0);
  });

  form.querySelector('button[type="submit"]').disabled = true;

  try {
    const response = await fetch('/api/test_single_prediction', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Unknown error');
    }

    renderPrediction(result);
    renderComparison(result);
    renderSpiderChart(payload);
  } catch (err) {
    console.error(err);
    renderPrediction({ error: err.message });
    const errNode = document.getElementById('form-error');
    errNode.textContent = err.message;
    errNode.classList.remove('hidden');
  } finally {
    form.querySelector('button[type="submit"]').disabled = false;
  }
}

function attachFormHandler() {
  const form = document.getElementById('prediction-form');
  if (!form) return;
  form.addEventListener('submit', submitPrediction);
}

let spiderChartInstance = null;

function renderSpiderChart(metricsData) {
  const card = document.getElementById('spider-chart-card');
  card.classList.remove('hidden');
  
  const ctx = document.getElementById('spider-chart');
  
  // Destroy existing chart if it exists
  if (spiderChartInstance) {
    spiderChartInstance.destroy();
  }
  
  const labels = [
    'Network In (Mbps)',
    'Network Out (Mbps)',
    'Response Time (ms)',
    'CPU Utilization (%)',
    'Memory Utilization (%)',
    'Disk Usage (%)'
  ];
  
  const values = [
    metricsData.Network_In_Mbps || 0,
    metricsData.Network_Out_Mbps || 0,
    metricsData.Response_Time_ms || 0,
    metricsData.CPU_Utilization_Percent || 0,
    metricsData.Memory_Utilization_Percent || 0,
    metricsData.Disk_Usage_Percent || 0
  ];
  
  // Use fixed max of 100 for consistent scaling
  const suggestedMax = 100;
  
  spiderChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Workload Metrics',
        data: values,
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        borderColor: 'rgba(34, 197, 94, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(34, 197, 94, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(34, 197, 94, 1)',
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          suggestedMax: suggestedMax,
          ticks: {
            color: '#64748b',
            font: {
              size: 10
            },
            stepSize: suggestedMax / 5,
            backdropColor: 'transparent'
          },
          grid: {
            color: '#e2e8f0',
            lineWidth: 1
          },
          angleLines: {
            color: '#cbd5e1',
            lineWidth: 1
          },
          pointLabels: {
            color: '#475569',
            font: {
              size: 12,
              weight: '500'
            },
            padding: 8
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: 12,
          titleFont: {
            size: 13
          },
          bodyFont: {
            size: 12
          },
          callbacks: {
            label: function(context) {
              const label = context.label || '';
              const value = context.parsed.r || 0;
              
              // Format based on metric type
              if (label.includes('Mbps')) {
                return `${value.toFixed(2)} Mbps`;
              } else if (label.includes('ms')) {
                return `${value.toFixed(2)} ms`;
              } else {
                return `${value.toFixed(1)}%`;
              }
            }
          }
        }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  await loadStandardMetrics();
  attachFormHandler();
});
