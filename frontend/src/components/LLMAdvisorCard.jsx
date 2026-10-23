import React, { useState, useEffect } from 'react';
import { Sparkles, AlertTriangle, Wrench, ShieldCheck, RefreshCw, CheckCircle2 } from 'lucide-react';
import { fetchLLMAdvice } from '../services/api';

export default function LLMAdvisorCard({
  machineType = 'pump',
  analysisResult = null,
}) {
  const [adviceData, setAdviceData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const isAnomaly = analysisResult?.is_anomaly ?? (analysisResult?.score >= analysisResult?.threshold);
  const score = analysisResult?.score || 0;
  const threshold = analysisResult?.threshold || 0.0201;
  const zScore = analysisResult?.dynamic_metrics?.z_score || 0;
  const primaryDefect = analysisResult?.primary_defect_band || 'Uniformly Nominal';
  const bands = analysisResult?.frequency_bands || [];

  const loadAdvice = async () => {
    if (!analysisResult) return;
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetchLLMAdvice({
        machineType,
        score,
        threshold,
        zScore,
        isAnomaly,
        primaryDefect,
        frequencyBands: bands,
      });
      if (res && res.advice) {
        setAdviceData(res);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg('Failed to generate LLM advice. Please check backend connection.');
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-fetch advice whenever a new analysisResult arrives
  useEffect(() => {
    if (analysisResult) {
      loadAdvice();
    } else {
      setAdviceData(null);
    }
  }, [analysisResult, machineType]);

  if (!analysisResult) {
    return null;
  }

  const advice = adviceData?.advice;
  const isGeminiLive = adviceData?.source === 'gemini';

  return (
    <div className="glass-panel llm-advisor-container" style={{ marginTop: '24px' }}>
      {/* Header */}
      <div className="card-header" style={{ flexWrap: 'wrap', gap: '10px' }}>
        <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sparkles size={18} color="var(--accent-cyan)" />
          Gemini 3.5 Flash Industrial Advisor
        </h2>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span
            style={{
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              padding: '3px 8px',
              borderRadius: '9999px',
              background: isGeminiLive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(56, 189, 248, 0.12)',
              color: isGeminiLive ? 'var(--success)' : 'var(--accent-cyan)',
              border: `1px solid ${isGeminiLive ? 'rgba(16, 185, 129, 0.3)' : 'rgba(56, 189, 248, 0.25)'}`,
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
            }}
          >
            <span className="pulse-dot" style={{ background: isGeminiLive ? 'var(--success)' : 'var(--accent-cyan)' }} />
            {isGeminiLive ? adviceData?.model_used || 'Gemini 3.5 Flash' : 'Gemini 3.5 Flash'}
          </span>

          <button
            type="button"
            className="btn-refresh-llm"
            disabled={isLoading}
            onClick={() => loadAdvice()}
            title="Re-run Gemini Analysis"
          >
            <RefreshCw size={13} className={isLoading ? 'spin-icon' : ''} />
            {isLoading ? 'Analyzing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Summary Banner */}
      {advice?.summary && (
        <div className={`llm-summary-banner ${isAnomaly ? 'anomaly' : 'normal'}`}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '13px', lineHeight: 1.5, color: '#f1f5f9' }}>
            {isAnomaly ? (
              <AlertTriangle size={17} color="var(--danger)" style={{ flexShrink: 0, marginTop: '2px' }} />
            ) : (
              <CheckCircle2 size={17} color="var(--success)" style={{ flexShrink: 0, marginTop: '2px' }} />
            )}
            <div>
              <strong>Executive AI Diagnosis:</strong> {advice.summary}
            </div>
          </div>
        </div>
      )}

      {/* Anomaly Analysis: ONLY shown when anomaly is present */}
      {isAnomaly && advice && (advice.root_causes?.length > 0 || advice.possible_solutions?.length > 0 || advice.preventive_measures?.length > 0) && (
        <div className="llm-advice-grid">
          {/* 1. Root Causes */}
          <div className="advice-pillar pillar-causes">
            <div className="pillar-header">
              <AlertTriangle size={16} color="var(--danger)" />
              <span>Root Causes for Anomaly</span>
            </div>
            <p className="pillar-subtitle">Primary mechanical, fluid, or electrical drivers of the defect:</p>
            <ul className="pillar-list">
              {advice.root_causes?.map((item, idx) => (
                <li key={idx} className="pillar-list-item">
                  <span className="bullet-indicator bullet-causes" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* 2. Possible Solutions */}
          <div className="advice-pillar pillar-solutions">
            <div className="pillar-header">
              <Wrench size={16} color="var(--accent-cyan)" />
              <span>Possible Immediate Solutions</span>
            </div>
            <p className="pillar-subtitle">Actionable corrective steps for technicians on the plant floor:</p>
            <ul className="pillar-list">
              {advice.possible_solutions?.map((item, idx) => (
                <li key={idx} className="pillar-list-item">
                  <span className="bullet-indicator bullet-solutions" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* 3. Preventive Measures */}
          <div className="advice-pillar pillar-prevention">
            <div className="pillar-header">
              <ShieldCheck size={16} color="var(--success)" />
              <span>Preventive Measures</span>
            </div>
            <p className="pillar-subtitle">Condition monitoring and maintenance protocols to avoid recurrence:</p>
            <ul className="pillar-list">
              {advice.preventive_measures?.map((item, idx) => (
                <li key={idx} className="pillar-list-item">
                  <span className="bullet-indicator bullet-prevention" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Normal Operating State Reassurance: Rendered when data is nominal */}
      {!isAnomaly && advice && (
        <div className="nominal-status-panel">
          <div className="nominal-status-header">
            <CheckCircle2 size={16} color="var(--success)" />
            <span>Operational Status: Healthy &amp; Nominal</span>
          </div>
          <p className="nominal-status-desc">
            All acoustic reconstruction residuals and frequency bands remain well below adaptive threshold limits. No anomalous mechanical vibration, bearing friction, or fluid defects detected. Asset is operating optimally.
          </p>
          <div className="nominal-metrics-row">
            <div className="nominal-metric-chip">
              <span className="chip-label">Acoustic Residual</span>
              <span className="chip-val chip-success">{Number(score).toFixed(4)} MSE</span>
            </div>
            <div className="nominal-metric-chip">
              <span className="chip-label">Adaptive Limit</span>
              <span className="chip-val">{Number(threshold).toFixed(4)}</span>
            </div>
            <div className="nominal-metric-chip">
              <span className="chip-label">Statistical Deviation</span>
              <span className="chip-val chip-success">{Number(zScore) > 0 ? `+${Number(zScore).toFixed(2)}σ` : `${Number(zScore).toFixed(2)}σ`}</span>
            </div>
            <div className="nominal-metric-chip">
              <span className="chip-label">Maintenance Directive</span>
              <span className="chip-val chip-success">Cleared for Normal Operation</span>
            </div>
          </div>
        </div>
      )}

      {errorMsg && (
        <div style={{ color: 'var(--danger)', fontSize: '12px', marginTop: '12px', textAlign: 'center' }}>
          {errorMsg}
        </div>
      )}
    </div>
  );
}
