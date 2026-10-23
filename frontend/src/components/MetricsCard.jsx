import React from 'react';
import { Cpu, HardDrive, Zap, Layers } from 'lucide-react';

export default function MetricsCard({ profile }) {
  if (!profile) return null;

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="metric-label">Model Size (Flash)</div>
          <HardDrive size={16} color="var(--accent-cyan)" />
        </div>
        <div className="metric-value">{profile.model_size_kb} KB</div>
        <div className="metric-sub" style={{ color: profile.within_size_budget ? 'var(--success)' : 'var(--danger)' }}>
          {profile.within_size_budget ? '✅ Under 256 KB Budget' : '⚠️ Exceeds Budget'}
        </div>
      </div>

      <div className="metric-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="metric-label">Inference Latency</div>
          <Zap size={16} color="var(--warning)" />
        </div>
        <div className="metric-value">{profile.avg_latency_ms} ms</div>
        <div className="metric-sub">Real-time edge response</div>
      </div>

      <div className="metric-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="metric-label">Peak RAM (SRAM)</div>
          <Cpu size={16} color="var(--accent-purple)" />
        </div>
        <div className="metric-value">{profile.peak_ram_kb} KB</div>
        <div className="metric-sub">Tensor arena buffer</div>
      </div>

      <div className="metric-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="metric-label">Quantization</div>
          <Layers size={16} color="var(--accent-blue)" />
        </div>
        <div className="metric-value" style={{ fontSize: '18px', paddingTop: '4px' }}>
          {profile.quantization || 'INT8'}
        </div>
        <div className="metric-sub">{profile.hardware_target || 'Edge MCU'}</div>
      </div>
    </div>
  );
}
