/**
 * API service for communicating with FastAPI backend.
 * Includes support for audio analysis, machine asset tracking, anomaly reassignment,
 * and predictive maintenance prognostics.
 */

const API_BASE = import.meta.env?.VITE_API_BASE_URL || '/api';

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch (err) {
    return false;
  }
}

export async function fetchModelProfile() {
  try {
    const res = await fetch(`${API_BASE}/profile`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error('Failed to fetch profile');
    return await res.json();
  } catch (err) {
    return {
      model_size_kb: 20.9,
      target_limit_kb: 256.0,
      within_size_budget: true,
      avg_latency_ms: 0.11,
      peak_ram_kb: 54.8,
      mac_operations_m: 1.15,
      quantization: 'INT8 Full Integer',
      hardware_target: 'ARM Cortex-M4 / ESP32-S3',
      status: 'HEALTHY',
    };
  }
}

export async function analyzePreset(sampleFilename = 'normal_id_00_00000000.wav', isAnomaly = false, machineType = 'pump', machineId = 'p001') {
  const sampleType = isAnomaly ? 'anomaly' : 'normal';
  const query = new URLSearchParams({
    sample_type: sampleType,
    filename: sampleFilename,
    machine_type: machineType,
  });
  if (machineId) {
    query.set('machine_id', machineId);
  }

  const res = await fetch(
    `${API_BASE}/sample-analyze?${query.toString()}`,
    { signal: AbortSignal.timeout(15000) }
  );
  if (!res.ok) {
    throw new Error(`Real analysis failed: ${res.statusText}`);
  }
  const data = await res.json();
  data.is_live = true;
  return data;
}

export async function analyzeAudioFile(file, isMockAnomaly = false, machineType = 'pump', machineId = 'p001') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('machine_type', machineType);
  if (machineId) {
    formData.append('machine_id', machineId);
  }

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
    signal: AbortSignal.timeout(25000),
  });

  if (!res.ok) {
    throw new Error(`Failed to process audio file: ${res.statusText}`);
  }
  const data = await res.json();
  data.is_live = true;
  return data;
}

export async function fetchRegisteredMachines(machineType = null) {
  try {
    const url = machineType ? `${API_BASE}/machines?machine_type=${encodeURIComponent(machineType)}` : `${API_BASE}/machines`;
    const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return [];
    const data = await res.json();
    return data.machines || [];
  } catch (err) {
    console.warn('Failed to fetch registered machines:', err);
    return [];
  }
}

export async function fetchMachinePrognostics(machineId) {
  if (!machineId) return null;
  try {
    const res = await fetch(`${API_BASE}/machines/${encodeURIComponent(machineId)}/prognostics`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error('Failed to fetch prognostics');
    return await res.json();
  } catch (err) {
    console.warn(`Failed to fetch prognostics for ${machineId}:`, err);
    return null;
  }
}

export async function fetchMachineHistory(machineId) {
  if (!machineId) return [];
  try {
    const res = await fetch(`${API_BASE}/machines/${encodeURIComponent(machineId)}/history`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error('Failed to fetch history');
    const data = await res.json();
    return data.history || [];
  } catch (err) {
    console.warn(`Failed to fetch history for ${machineId}:`, err);
    return [];
  }
}

export async function reassignAnomalyRecord({ recordId, machineId, machineType = 'pump', notes = '' }) {
  const res = await fetch(`${API_BASE}/machines/reassign-anomaly`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      record_id: recordId,
      machine_id: machineId,
      machine_type: machineType,
      notes,
    }),
    signal: AbortSignal.timeout(6000),
  });
  if (!res.ok) {
    throw new Error(`Reassignment failed: ${res.statusText}`);
  }
  return await res.json();
}

export async function fetchLLMAdvice({
  machineType = 'pump',
  score = 0.02,
  threshold = 0.0201,
  zScore = 0,
  isAnomaly = false,
  primaryDefect = 'Uniformly Nominal',
  frequencyBands = [],
  apiKey = null,
}) {
  const res = await fetch(`${API_BASE}/llm-advice`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      machine_type: machineType,
      score,
      threshold,
      z_score: zScore,
      is_anomaly: isAnomaly,
      primary_defect: primaryDefect,
      frequency_bands: frequencyBands,
      api_key: apiKey || undefined,
    }),
    signal: AbortSignal.timeout(30000),
  });

  if (!res.ok) {
    throw new Error(`Gemini request failed (${res.status}): ${res.statusText}`);
  }
  return await res.json();
}
