let cachedResults = [];
let filteredResults = [];
let geminiCache = {};
let activeMonthLabel;
let realtimeErrorNode;

function formatCurrency(value) {
  if (!Number.isFinite(value)) return '$0.00';
  return `$${value.toFixed(2)}`;
}

function renderSummary(payload) {
  const root = document.getElementById('summary-cards');
  const total = document.getElementById('summary-total');
  const optimal = document.getElementById('summary-optimal');
  const over = document.getElementById('summary-over');
  const under = document.getElementById('summary-under');
  const savings = document.getElementById('summary-savings');

  const totalInstances = payload.total_instances ?? 0;
  total.textContent = totalInstances;
  optimal.textContent = payload.status_summary?.optimal ?? 0;
  over.textContent = payload.status_summary?.['over-provisioned'] ?? 0;
  under.textContent = payload.status_summary?.['under-provisioned'] ?? 0;
  
  // Calculate total current waste (over-provisioned = negative)
  const totalCurrentWaste = payload.results?.reduce((sum, item) => {
    const monthlySavings = Number(item.analysis?.potential_monthly_savings) || 0;
    const status = item.analysis?.status;
    // Over-provisioned instances show current waste (negative)
    return sum + (status === 'over-provisioned' && monthlySavings > 0 ? monthlySavings : 0);
  }, 0) || 0;
  
  // Calculate total improvement opportunity (under-provisioned = positive)
  const totalImprovementCost = payload.results?.reduce((sum, item) => {
    const monthlySavings = Number(item.analysis?.potential_monthly_savings) || 0;
    const status = item.analysis?.status;
    // Under-provisioned instances show opportunity cost (positive)
    return sum + (status === 'under-provisioned' && monthlySavings < 0 ? Math.abs(monthlySavings) : 0);
  }, 0) || 0;
  
  // Net = improvement opportunity - current waste
  const netSavings = totalImprovementCost - totalCurrentWaste;
  savings.textContent = formatCurrency(Math.abs(netSavings));
  
  // Color code based on net result
  if (netSavings > 0) {
    savings.className = 'text-3xl font-semibold mt-2 text-green-900';
  } else if (netSavings < 0) {
    savings.className = 'text-3xl font-semibold mt-2 text-red-900';
  } else {
    savings.className = 'text-3xl font-semibold mt-2 text-gray-900';
  }

  root.hidden = totalInstances === 0;
}

function renderTableRows(results) {
  const tbody = document.querySelector('#results-table tbody');
  
  // Use DocumentFragment for better performance
  const fragment = document.createDocumentFragment();

  results.forEach((item, index) => {
    const analysis = item.analysis ?? {};
    // Get recommended type from Gemini (preferred) or LSTM (fallback)
    const recommendedType = item.gemini_prediction?.instance_type || item.lstm_prediction?.instance_type || 'unknown';
    const monthlySavings = Number(analysis.potential_monthly_savings) || 0;
    const status = analysis.status || 'unknown';
    
    // Determine display text and color based on status and savings
    // User requirement: under-provisioned = positive (+), over-provisioned = negative (-)
    let displayText = '$0.00';
    let displayColor = 'text-gray-600';
    
    if (status === 'optimal') {
      // Optimal = no change needed
      displayText = '$0.00';
      displayColor = 'text-gray-600';
    } else if (status === 'over-provisioned') {
      // Over-provisioned = wasting money NOW (show as negative/cost)
      displayText = monthlySavings > 0 ? `-${formatCurrency(monthlySavings)}` : `${formatCurrency(Math.abs(monthlySavings))}`;
      displayColor = 'text-red-600'; // Red for current waste
    } else if (status === 'under-provisioned') {
      // Under-provisioned = opportunity to improve (show as positive/savings potential)
      displayText = monthlySavings < 0 ? `+${formatCurrency(Math.abs(monthlySavings))}` : `${formatCurrency(monthlySavings)}`;
      displayColor = 'text-green-600'; // Green for improvement opportunity
    } else {
      // Unknown status
      if (monthlySavings > 0) {
        displayText = `-${formatCurrency(monthlySavings)}`;
        displayColor = 'text-red-600';
      } else if (monthlySavings < 0) {
        displayText = `+${formatCurrency(Math.abs(monthlySavings))}`;
        displayColor = 'text-green-600';
      }
    }
    
    const row = document.createElement('tr');
    row.className = 'hover:bg-gray-50 transition';
    row.dataset.instanceId = item.instance_id;
    row.dataset.status = status;
    row.innerHTML = `
      <td class="py-3 pr-4 pl-4 font-mono text-xs text-gray-700">${item.instance_id}</td>
      <td class="py-3 pr-4 text-gray-800">${item.current_type}</td>
      <td class="py-3 pr-4 text-green-700 font-semibold">${recommendedType}</td>
      <td class="py-3 pr-4">
        <span class="px-3 py-1 rounded-full text-xs uppercase tracking-wide font-semibold ${statusBadgeClass(status)}">${status}</span>
      </td>
      <td class="py-3 pr-4">
        <span class="${displayColor} font-semibold">${displayText}</span>
      </td>
      <td class="py-3 pr-4 text-center">
        <div class="flex items-center justify-center gap-2">
          <button data-index="${index}" class="view-instance inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white transition text-xs uppercase tracking-wide font-semibold">
            <i class="fas fa-eye"></i>
            Inspect
          </button>
          <button class="execute-btn inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition text-xs uppercase tracking-wide font-semibold">
            <i class="fas fa-play"></i>
            Execute
          </button>
        </div>
      </td>
    `;
    fragment.appendChild(row);
  });
  
  // Single DOM update for better performance
  tbody.innerHTML = '';
  tbody.appendChild(fragment);
  attachRowListeners();
}

function determineCostStatus(instance) {
  const analysis = instance.analysis ?? {};
  const status = analysis.status;
  const monthlySavings = Number(analysis.potential_monthly_savings) || 0;
  
  if (status === 'over-provisioned' && monthlySavings > 0) {
    return { color: 'text-green-600', amount: monthlySavings };
  } else if (status === 'under-provisioned' && monthlySavings < 0) {
    // Under-provisioned means additional cost (negative savings)
    return { color: 'text-red-600', amount: Math.abs(monthlySavings) };
  } else if (status === 'over-provisioned' && monthlySavings < 0) {
    return { color: 'text-amber-600', amount: Math.abs(monthlySavings) };
  } else {
    // Optimal or zero savings
    return { color: 'text-gray-600', amount: Math.abs(monthlySavings) };
  }
}

function statusBadgeClass(status) {
  switch (status) {
    case 'over-provisioned':
      return 'bg-red-50 border border-red-200 text-red-700';
    case 'under-provisioned':
      return 'bg-amber-50 border border-amber-200 text-amber-700';
    case 'optimal':
      return 'bg-green-50 border border-green-200 text-green-700';
    default:
      return 'bg-gray-50 border border-gray-200 text-gray-700';
  }
}

function populateSelectorOptions(instances) {
  if (!instanceSelector) return;
  instanceSelector.innerHTML = '';
  if (!instances.length) {
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'No recommendations available';
    instanceSelector.appendChild(placeholder);
    instanceSelector.disabled = true;
    return;
  }
  instanceSelector.disabled = false;
  instances.forEach((item, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    const recommendedType = item.gemini_prediction?.instance_type || item.lstm_prediction?.instance_type || 'unknown';
    option.textContent = `${item.instance_id} → ${recommendedType}`;
    instanceSelector.appendChild(option);
  });
}

async function openModal(instance, index) {
  const modal = document.getElementById('comparison-modal');
  const title = document.getElementById('modal-title');
  const subtitle = document.getElementById('modal-subtitle');
  const statusBadge = document.getElementById('modal-status-badge');
  const specTableBody = document.querySelector('#modal-spec-table tbody');
  const specDisclaimer = document.getElementById('modal-spec-disclaimer');
  const recommendations = document.getElementById('modal-recommendations');
  const summary = document.getElementById('modal-summary');

  if (!instance) return;

  const analysis = instance.analysis ?? {};
  const costStatus = determineCostStatus(instance);
  const recommendedType = instance.gemini_prediction?.instance_type || instance.lstm_prediction?.instance_type || 'unknown';

  title.textContent = `${instance.instance_id}`;
  subtitle.textContent = `${instance.current_type} → ${recommendedType}`;
  statusBadge.textContent = analysis.status;
  statusBadge.className = `px-3 py-1 rounded-full text-xs uppercase tracking-wide font-semibold ${statusBadgeClass(analysis.status)}`;

  // Show modal
  modal.classList.add('active');

  // Clear previous data
  specTableBody.innerHTML = '<tr><td colspan="4" class="py-6 text-center text-gray-500">Loading specifications...</td></tr>';
  recommendations.innerHTML = '';
  summary.innerHTML = '';

  // Build summary with both predictions
  const lstmType = instance.lstm_prediction?.instance_type || 'N/A';
  const lstmConfidence = instance.lstm_prediction?.confidence || 0;
  const geminiType = instance.gemini_prediction?.instance_type || 'N/A';
  const geminiConfidence = instance.gemini_prediction?.confidence || 'unknown';
  const modelsAgree = instance.models_agree;
  
  const summaryBlocks = [];
  
  // Current instance info
  summaryBlocks.push({
    label: 'Current Instance',
    value: `<span class="text-gray-900">${instance.current_type}</span>`,
    extra: modelsAgree ? 
      '<span class="text-xs text-green-600 font-semibold">✓ Models Agree</span>' :
      '<span class="text-xs text-amber-600 font-semibold">⚠ Models Disagree</span>'
  });
  
  // LSTM prediction
  summaryBlocks.push({
    label: 'LSTM Model',
    value: `<span class="text-blue-700">${lstmType}</span>`,
    extra: `<span class="text-xs text-gray-600">Confidence: ${lstmConfidence.toFixed(1)}%</span>`
  });
  
  // Gemini prediction
  summaryBlocks.push({
    label: 'Gemini AI',
    value: `<span class="text-green-700">${geminiType}</span>`,
    extra: `<span class="text-xs text-gray-600 capitalize">Confidence: ${geminiConfidence}</span>`
  });
  
  // Cost information - Company perspective
  if (Number.isFinite(analysis.potential_monthly_savings)) {
    const savingsAmount = analysis.potential_monthly_savings;
    const status = analysis.status;
    
    let displayValue, displayColor, displayLabel;
    
    if (status === 'optimal') {
      displayValue = '$0.00';
      displayColor = 'text-gray-600';
      displayLabel = '';
    } else if (status === 'over-provisioned') {
      // Over-provisioned = current waste (negative/red)
      displayValue = savingsAmount > 0 ? `-${formatCurrency(savingsAmount)}` : formatCurrency(Math.abs(savingsAmount));
      displayColor = 'text-red-600';
      displayLabel = '<span class="text-xs text-red-600">Current Waste</span>';
    } else if (status === 'under-provisioned') {
      // Under-provisioned = investment opportunity (positive/green)
      displayValue = savingsAmount < 0 ? `+${formatCurrency(Math.abs(savingsAmount))}` : formatCurrency(savingsAmount);
      displayColor = 'text-green-600';
      displayLabel = '<span class="text-xs text-green-600">Investment Opportunity</span>';
    } else {
      displayValue = formatCurrency(Math.abs(savingsAmount));
      displayColor = savingsAmount > 0 ? 'text-red-600' : 'text-green-600';
      displayLabel = '';
    }
    
    summaryBlocks.push({
      label: 'Monthly Impact',
      value: `<span class="${displayColor}">${displayValue}</span>`,
      extra: displayLabel
    });
  }

  if (summaryBlocks.length) {
    summary.innerHTML = summaryBlocks
      .map(
        (item) => `
          <div class="rounded-lg bg-gray-50 border border-gray-200 p-4">
            <p class="text-xs uppercase tracking-wide text-gray-500">${item.label}</p>
            <p class="mt-2 text-lg font-semibold">${item.value}</p>
            ${item.extra ? `<p class="mt-1">${item.extra}</p>` : ''}
          </div>
        `
      )
      .join('');
  }

  try {
    const specData = await fetchGeminiDetails(instance);
    
    if (specData?.rows?.length) {
      // Add Monthly Impact row at the end with highlighting
      const monthlySavings = Number(analysis.potential_monthly_savings) || 0;
      const status = analysis.status;
      
      let displayValue, displayUnit;
      
      if (status === 'optimal') {
        displayValue = '$0.00';
        displayUnit = '';
      } else if (status === 'over-provisioned') {
        // Over-provisioned = wasting money (negative/red)
        displayValue = monthlySavings > 0 ? `-${formatCurrency(monthlySavings)}` : formatCurrency(Math.abs(monthlySavings));
        displayUnit = 'current waste/month';
      } else if (status === 'under-provisioned') {
        // Under-provisioned = investment needed (positive/green)
        displayValue = monthlySavings < 0 ? `+${formatCurrency(Math.abs(monthlySavings))}` : formatCurrency(monthlySavings);
        displayUnit = 'investment/month';
      } else {
        displayValue = formatCurrency(Math.abs(monthlySavings));
        displayUnit = 'per month';
      }
      
      const costSavingRow = {
        metric: 'Monthly Impact',
        current: '-',
        recommended: displayValue,
        unit: displayUnit
      };
      
      // Gemini rows first, then monthly savings at the end
      const allRows = [...specData.rows, costSavingRow];
      
      specTableBody.innerHTML = allRows
        .map((row, idx) => {
          const current = row.current ?? '—';
          const recommended = row.recommended ?? '—';
          const unit = row.unit ?? '';
          const isLastRow = idx === allRows.length - 1;
          const textColor = isLastRow ? costStatus.color : 'text-gray-700';
          const bgColor = isLastRow ? 'bg-yellow-50 border-t-2 border-yellow-300' : '';
          const fontWeight = isLastRow ? 'font-bold' : 'font-medium';
          
          return `
            <tr class="${bgColor}">
              <td class="py-3 pr-4 pl-4 text-gray-900 ${fontWeight}">${row.metric}</td>
              <td class="py-3 pr-4 ${textColor} ${isLastRow ? 'font-semibold' : ''}">${current}</td>
              <td class="py-3 pr-4 ${textColor} font-semibold ${isLastRow ? 'text-lg' : ''}">${recommended}</td>
              <td class="py-3 pr-4 text-gray-600 ${isLastRow ? 'font-semibold' : ''}">${unit}</td>
            </tr>
          `;
        })
        .join('');
      specDisclaimer.textContent = specData.disclaimer ?? '';
    } else {
      specTableBody.innerHTML = '<tr><td colspan="4" class="py-4 text-center text-gray-500">No specification data available</td></tr>';
    }

    const recs = Array.isArray(specData?.recommendations)
      ? specData.recommendations.filter((item) => typeof item === 'string' && item.trim().length > 0)
      : [];

    if (recs.length) {
      recommendations.innerHTML = recs.map((text) => `<li class="text-gray-700">${text}</li>`).join('');
    } else {
      recommendations.innerHTML = '<li class="text-gray-500">No additional recommendations available</li>';
    }
  } catch (error) {
    console.error('Gemini comparison failed', error);
    specTableBody.innerHTML = `
      <tr>
        <td colspan="4" class="py-3 text-center text-red-600">${error.message ?? 'Unable to retrieve Gemini metrics.'}</td>
      </tr>
    `;
    specDisclaimer.textContent = 'Check backend logs or API quota.';
    recommendations.innerHTML = '<li class="text-gray-500">Gemini guidance unavailable</li>';
  }
}

function closeModal() {
  const modal = document.getElementById('comparison-modal');
  modal.classList.remove('active');
}

async function renderComparison(instance, index) {
  // This function is now replaced by openModal
  await openModal(instance, index);
}

function attachRowListeners() {
  document.querySelectorAll('.view-instance').forEach((button) => {
    button.addEventListener('click', (event) => {
      const index = Number(event.currentTarget.getAttribute('data-index'));
      const instance = cachedResults[index];
      openModal(instance, index);
    });
  });
}

async function fetchGeminiDetails(instance) {
  const predictedType = instance.gemini_prediction?.instance_type || instance.lstm_prediction?.instance_type || 'unknown';
  const cacheKey = `${instance.instance_id}|${predictedType}`;
  if (geminiCache[cacheKey]) {
    return geminiCache[cacheKey];
  }

  const response = await fetch('/api/gemini_metrics', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      instance_id: instance.instance_id,
      current_type: instance.current_type,
      predicted_type: predictedType,
      status: instance.analysis?.status,
      metrics: instance.analysis?.metrics_comparison || {},
    }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || 'Gemini API request failed');
  }

  const data = await response.json();
  geminiCache[cacheKey] = data;
  return data;
}

async function fetchRealtimeAnalysis() {
  const resultsCard = document.getElementById('results-card');

  realtimeErrorNode?.classList.add('hidden');
  if (activeMonthLabel) {
    activeMonthLabel.textContent = 'Loading previous month telemetry…';
  }

  try {
    const response = await fetch('/api/analyze_month', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || 'Unable to analyze data');
    }

    cachedResults = payload.results ?? [];
    filteredResults = [...cachedResults];
    geminiCache = {};

    const activeLabel = payload.month || payload.active_month || 'Previous month';
    if (activeMonthLabel) {
      activeMonthLabel.textContent = activeLabel;
    }

    if (cachedResults.length === 0) {
      renderSummary({ total_instances: 0, status_summary: {}, total_potential_savings: 0, results: [] });
      resultsCard.classList.add('hidden');
      if (realtimeErrorNode) {
        realtimeErrorNode.textContent = 'No recommendations available for this period.';
        realtimeErrorNode.classList.remove('hidden');
      }
      return;
    }

    renderSummary(payload);
    renderTableRows(filteredResults);

    resultsCard.classList.remove('hidden');
  } catch (error) {
    console.error(error);
    if (realtimeErrorNode) {
      realtimeErrorNode.textContent = error.message || 'Unable to analyze data';
      realtimeErrorNode.classList.remove('hidden');
    }
    resultsCard?.classList.add('hidden');
    document.getElementById('summary-cards').hidden = true;
    if (activeMonthLabel) {
      activeMonthLabel.textContent = 'Previous month';
    }
  }
}

function applyFilters() {
  const searchInput = document.getElementById('search-input');
  const statusFilter = document.getElementById('status-filter');
  
  const searchTerm = searchInput?.value.toLowerCase().trim() || '';
  const statusValue = statusFilter?.value || 'all';
  
  filteredResults = cachedResults.filter(item => {
    const matchesSearch = !searchTerm || item.instance_id.toLowerCase().includes(searchTerm);
    const matchesStatus = statusValue === 'all' || item.analysis.status === statusValue;
    return matchesSearch && matchesStatus;
  });
  
  renderTableRows(filteredResults);
}

document.addEventListener('DOMContentLoaded', () => {
  activeMonthLabel = document.getElementById('active-month-label');
  realtimeErrorNode = document.getElementById('realtime-error');
  
  // Search and filter event listeners
  const searchInput = document.getElementById('search-input');
  const statusFilter = document.getElementById('status-filter');
  
  if (searchInput) {
    searchInput.addEventListener('input', applyFilters);
  }
  
  if (statusFilter) {
    statusFilter.addEventListener('change', applyFilters);
  }
  
  // Close modal when clicking backdrop
  document.getElementById('comparison-modal')?.addEventListener('click', (event) => {
    if (event.target.id === 'comparison-modal') {
      closeModal();
    }
  });

  fetchRealtimeAnalysis();
});
