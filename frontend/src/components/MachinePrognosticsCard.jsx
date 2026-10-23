import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  AlertTriangle,
  Clock,
  Activity,
  ShieldAlert,
  CheckCircle2,
  Calendar,
  Layers,
  Wrench,
  HelpCircle,
  FileText,
  ChevronRight,
  Flame,
} from 'lucide-react';
import { fetchMachinePrognostics } from '../services/api';

export default function MachinePrognosticsCard({ machineId, machineType, refreshTrigger = 0 }) {
  const [prognostics, setPrognostics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('prediction'); // 'prediction' or 'history'

  useEffect(() => {
    async function load() {
      if (!machineId) return;
      setLoading(true);
      const data = await fetchMachinePrognostics(machineId);
      setPrognostics(data);
      setLoading(false);
    }
    load();
  }, [machineId, machineType, refreshTrigger]);

  if (!machineId) return null;

  const riskBadgeStyle = (level) => {
    switch (level) {
      case 'CRITICAL':
        return {
          bg: 'rgba(239, 68, 68, 0.2)',
          border: 'var(--danger)',
          color: '#f87171',
          icon: <Flame size={14} />,
        };
      case 'ELEVATED':
        return {
          bg: 'rgba(245, 158, 11, 0.2)',
          border: 'var(--warning)',
          color: '#fbbf24',
          icon: <AlertTriangle size={14} />,
        };
      case 'MODERATE':
        return {
          bg: 'rgba(56, 189, 248, 0.15)',
          border: 'var(--accent-cyan)',
          color: 'var(--accent-cyan)',
          icon: <Activity size={14} />,
        };
      default:
        return {
          bg: 'rgba(16, 185, 129, 0.15)',
          border: 'var(--success)',
          color: 'var(--success)',
          icon: <CheckCircle2 size={14} />,
        };
    }
  };

  const riskStyle = riskBadgeStyle(prognostics?.risk_level);

  return (
    <div
      className="glass-panel"
      style={{
        marginTop: '24px',
        padding: '24px',
        border: prognostics?.risk_level === 'CRITICAL' ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid var(--border-color)',
        boxShadow: prognostics?.risk_level === 'CRITICAL' ? 'var(--shadow-glow-danger)' : 'var(--shadow-card)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          paddingBottom: '18px',
          borderBottom: '1px solid var(--border-color)',
          marginBottom: '18px',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span
              style={{
                fontSize: '11px',
                fontWeight: 700,
                textTransform: 'uppercase',
                color: 'var(--accent-cyan)',
                letterSpacing: '0.05em',
              }}
            >
              Asset Prognostics & Predictive Health
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>•</span>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>MongoDB Historical Model</span>
          </div>
          <h2 style={{ fontSize: '18px', fontWeight: 800, marginTop: '4px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            Machine Asset: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{machineId}</span>
            <span style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-muted)', textTransform: 'capitalize' }}>
              ({machineType})
            </span>
          </h2>
        </div>

        {/* Risk Level Badge */}
        {prognostics && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '20px',
              background: riskStyle.bg,
              border: `1px solid ${riskStyle.border}`,
              color: riskStyle.color,
              fontWeight: 800,
              fontSize: '12px',
              letterSpacing: '0.05em',
            }}
          >
            {riskStyle.icon}
            {prognostics.risk_level} PROGNOSTIC RISK
          </div>
        )}
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)', fontSize: '13px' }}>
          Computing historical fault pattern & recurrence frequency for {machineId}...
        </div>
      ) : !prognostics ? (
        <div style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
          No historical records found for this machine ID.
        </div>
      ) : (
        <>
          {/* Key Metric Tiles */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
              gap: '12px',
              marginBottom: '20px',
            }}
          >
            <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Total Inspections</div>
              <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                {prognostics.total_inspections}
              </div>
            </div>

            <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Recorded Anomalies</div>
              <div
                style={{
                  fontSize: '20px',
                  fontWeight: 700,
                  fontFamily: 'var(--font-mono)',
                  color: prognostics.anomaly_count > 0 ? 'var(--danger)' : 'var(--success)',
                  marginTop: '2px',
                }}
              >
                {prognostics.anomaly_count}
              </div>
            </div>

            <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Anomaly Rate</div>
              <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                {prognostics.anomaly_rate_pct}%
              </div>
            </div>

            <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Peak Anomaly Score</div>
              <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-purple)', marginTop: '2px' }}>
                {prognostics.peak_score?.toFixed(5)}
              </div>
            </div>
          </div>

          {/* Tab Navigation */}
          <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-color)', marginBottom: '16px' }}>
            <button
              type="button"
              onClick={() => setActiveTab('prediction')}
              style={{
                padding: '8px 16px',
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === 'prediction' ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                color: activeTab === 'prediction' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                fontWeight: 700,
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <Activity size={15} />
              🔮 Predictive Fault Prognosis ("What Can Probably Occur")
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('history')}
              style={{
                padding: '8px 16px',
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === 'history' ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                color: activeTab === 'history' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                fontWeight: 700,
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <Calendar size={15} />
              📋 Past Anomaly Records ({prognostics.history?.length || 0})
            </button>
          </div>

          {/* TAB 1: Predictive Fault Prognosis */}
          {activeTab === 'prediction' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* Prognosis Hero Card */}
              <div
                style={{
                  padding: '18px',
                  borderRadius: '10px',
                  background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%)',
                  border: '1px solid var(--border-glow)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '10px' }}>
                  <div>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-cyan)', textTransform: 'uppercase' }}>
                      Predicted Failure Mode for {machineId}
                    </span>
                    <h3 style={{ fontSize: '17px', fontWeight: 800, color: 'var(--text-primary)', marginTop: '4px' }}>
                      {prognostics.likely_failure_mode}
                    </h3>
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '6px 12px',
                      borderRadius: '8px',
                      background: 'rgba(239, 68, 68, 0.12)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      color: 'var(--danger)',
                      fontSize: '12px',
                      fontWeight: 700,
                    }}
                  >
                    <Clock size={14} />
                    Estimated Window: {prognostics.estimated_time_to_failure}
                  </div>
                </div>

                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '12px', lineHeight: '1.6' }}>
                  <strong>Probable Chain of Consequences:</strong> {prognostics.consequences}
                </p>
              </div>

              {/* Pattern & Recurrence Frequency Analysis */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  gap: '14px',
                }}
              >
                {/* Frequency Card */}
                <div style={{ padding: '14px', borderRadius: '8px', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-purple)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                    <Activity size={14} />
                    Recurrence Frequency Analysis
                  </div>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5', margin: 0 }}>
                    {prognostics.frequency_assessment}
                  </p>
                </div>

                {/* Score Trend Card */}
                <div style={{ padding: '14px', borderRadius: '8px', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                    <TrendingUp size={14} />
                    Degradation Trend & Cause
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                    <div>
                      <strong>Trajectory:</strong> <span style={{ color: 'var(--text-primary)' }}>{prognostics.score_trend}</span>
                    </div>
                    <div style={{ marginTop: '2px' }}>
                      <strong>Dominant Defect Band:</strong> <span style={{ color: 'var(--accent-cyan)' }}>{prognostics.dominant_defect_band}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Checklist */}
              <div
                style={{
                  padding: '16px',
                  borderRadius: '8px',
                  background: 'rgba(16, 185, 129, 0.05)',
                  border: '1px solid rgba(16, 185, 129, 0.2)',
                }}
              >
                <h4 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                  <Wrench size={15} />
                  Immediate Preventive Maintenance Recommendations:
                </h4>
                <ul style={{ margin: 0, paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {prognostics.preventive_actions?.map((act, idx) => (
                    <li key={idx} style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                      {act}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* TAB 2: Historical Anomaly Records */}
          {activeTab === 'history' && (
            <div>
              {prognostics.history?.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)', fontSize: '13px' }}>
                  No historical test records found for {machineId}.
                </div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table
                    style={{
                      width: '100%',
                      borderCollapse: 'collapse',
                      fontSize: '12px',
                      textAlign: 'left',
                    }}
                  >
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '8px 12px' }}>Timestamp (UTC)</th>
                        <th style={{ padding: '8px 12px' }}>Status</th>
                        <th style={{ padding: '8px 12px' }}>Anomaly Score</th>
                        <th style={{ padding: '8px 12px' }}>Defect Band</th>
                        <th style={{ padding: '8px 12px' }}>Maintenance Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prognostics.history.map((rec, idx) => (
                        <tr
                          key={idx}
                          style={{
                            borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                            background: rec.status === 'anomaly' ? 'rgba(239, 68, 68, 0.04)' : 'transparent',
                          }}
                        >
                          <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                            {rec.timestamp ? new Date(rec.timestamp).toLocaleString() : 'N/A'}
                          </td>
                          <td style={{ padding: '10px 12px' }}>
                            <span
                              style={{
                                padding: '2px 8px',
                                borderRadius: '4px',
                                fontWeight: 700,
                                fontSize: '11px',
                                background: rec.status === 'anomaly' ? 'var(--danger-bg)' : 'var(--success-bg)',
                                color: rec.status === 'anomaly' ? 'var(--danger)' : 'var(--success)',
                                border: `1px solid ${rec.status === 'anomaly' ? 'var(--danger-border)' : 'var(--success-border)'}`,
                              }}
                            >
                              {rec.status.toUpperCase()}
                            </span>
                          </td>
                          <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                            {rec.anomalyScore?.toFixed(5)}
                          </td>
                          <td style={{ padding: '10px 12px', color: 'var(--accent-cyan)' }}>
                            {rec.defectBand || 'Nominal Spectrum'}
                          </td>
                          <td style={{ padding: '10px 12px', color: 'var(--text-muted)', fontStyle: rec.notes ? 'normal' : 'italic' }}>
                            {rec.notes || 'Routine acoustic test'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
