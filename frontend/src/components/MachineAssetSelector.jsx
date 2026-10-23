import React, { useState, useEffect } from 'react';
import { Cpu, Plus, Check, AlertCircle, History, ShieldAlert, Sparkles, ChevronDown } from 'lucide-react';
import { fetchRegisteredMachines } from '../services/api';

const MACHINE_TYPES = [
  { id: 'pump', name: 'Pump', icon: '🚰', prefix: 'p' },
  { id: 'valve', name: 'Valve', icon: '⚙️', prefix: 'v' },
  { id: 'fan', name: 'Fan', icon: '🌀', prefix: 'f' },
];

export default function MachineAssetSelector({
  selectedMachineType,
  onMachineTypeChange,
  selectedMachineId,
  onMachineIdChange,
  refreshTrigger = 0,
}) {
  const [machines, setMachines] = useState([]);
  const [isCustomId, setIsCustomId] = useState(false);
  const [customInput, setCustomInput] = useState('');
  const [activeAssetStats, setActiveAssetStats] = useState(null);

  // Fetch registered machines from MongoDB whenever machine type changes or refreshed
  useEffect(() => {
    async function loadMachines() {
      const list = await fetchRegisteredMachines(selectedMachineType);
      setMachines(list);

      // If selectedMachineId is not in list and not custom, set default to first machine or p001
      if (!isCustomId) {
        const found = list.find((m) => m.machineId === selectedMachineId);
        if (found) {
          setActiveAssetStats(found);
        } else if (list.length > 0) {
          onMachineIdChange(list[0].machineId);
          setActiveAssetStats(list[0]);
        } else {
          const defaultId = `${selectedMachineType === 'pump' ? 'p' : selectedMachineType === 'valve' ? 'v' : 'f'}001`;
          onMachineIdChange(defaultId);
          setActiveAssetStats(null);
        }
      } else {
        const found = list.find((m) => m.machineId === selectedMachineId);
        setActiveAssetStats(found || null);
      }
    }
    loadMachines();
  }, [selectedMachineType, refreshTrigger]);

  // Update stats whenever selectedMachineId changes
  useEffect(() => {
    const found = machines.find((m) => m.machineId === selectedMachineId);
    setActiveAssetStats(found || null);
  }, [selectedMachineId, machines]);

  const handleTypeSelect = (typeId) => {
    onMachineTypeChange(typeId);
    setIsCustomId(false);
  };

  const handleDropdownSelect = (e) => {
    const val = e.target.value;
    if (val === '__new__') {
      setIsCustomId(true);
      const nextId = `${selectedMachineType === 'pump' ? 'p' : selectedMachineType === 'valve' ? 'v' : 'f'}00${machines.length + 1}`;
      setCustomInput(nextId);
      onMachineIdChange(nextId);
    } else {
      setIsCustomId(false);
      onMachineIdChange(val);
    }
  };

  const handleCustomInputSubmit = (e) => {
    e.preventDefault();
    if (customInput.trim()) {
      onMachineIdChange(customInput.trim());
    }
  };

  return (
    <div className="machine-asset-selector-wrapper">
      {/* Machine Type Selection */}
      <div className="selector-block">
        <label className="selector-label">
          <Cpu size={14} color="var(--accent-cyan)" />
          Step 1 · Choose Machine Asset Type
        </label>
        <div className="machine-cards-grid">
          {MACHINE_TYPES.map((m) => {
            const isSelected = m.id === selectedMachineType;
            return (
              <button
                key={m.id}
                type="button"
                className={`machine-select-card ${isSelected ? 'active' : ''}`}
                onClick={() => handleTypeSelect(m.id)}
              >
                <div className="machine-card-top">
                  <span className="machine-card-name">
                    <span>{m.icon}</span> {m.name}
                  </span>
                  <span className="machine-card-badge">
                    {m.id === 'pump' ? 'τ = 0.0201' : m.id === 'valve' ? 'τ = 0.0155' : 'τ = 0.0179'}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Machine ID Selection & Entry for Factory Maintenance Engineer */}
      <div className="selector-block" style={{ marginTop: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <label className="selector-label" style={{ marginBottom: 0 }}>
            <History size={14} color="var(--accent-purple)" />
            Step 1B · Machine Asset ID (Floor Tag)
          </label>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Each machine tracks its own history
          </span>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {/* Machine ID Dropdown */}
          <div style={{ position: 'relative', flex: 1 }}>
            <select
              className="form-control"
              style={{
                width: '100%',
                padding: '10px 14px',
                background: 'rgba(15, 23, 42, 0.85)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
              value={isCustomId ? '__new__' : selectedMachineId}
              onChange={handleDropdownSelect}
            >
              <optgroup label="Registered Active Assets in MongoDB">
                {machines.map((m) => (
                  <option key={m.machineId} value={m.machineId}>
                    {m.machineId} — ({m.anomaly_count} anomalies / {m.total_inspections} tests) • {m.latest_status.toUpperCase()}
                  </option>
                ))}
              </optgroup>
              <option value="__new__">➕ Enter / Register New Machine ID...</option>
            </select>
          </div>

          {/* If custom new ID is selected, show direct input */}
          {isCustomId && (
            <div style={{ display: 'flex', gap: '6px', flex: 1 }}>
              <input
                type="text"
                className="form-control"
                placeholder="e.g. p007, PUMP-LINE2-01"
                value={customInput}
                onChange={(e) => {
                  setCustomInput(e.target.value);
                  onMachineIdChange(e.target.value.trim());
                }}
                style={{
                  padding: '10px 14px',
                  background: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid var(--accent-cyan)',
                  borderRadius: '8px',
                  color: 'var(--accent-cyan)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '13px',
                  fontWeight: 600,
                  width: '100%',
                }}
              />
            </div>
          )}
        </div>

        {/* Active Machine Status Badge Bar */}
        <div
          style={{
            marginTop: '10px',
            padding: '8px 12px',
            background: 'rgba(255, 255, 255, 0.03)',
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: 'var(--text-muted)' }}>Target Asset:</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {selectedMachineId}
            </span>
            <span style={{ textTransform: 'capitalize', color: 'var(--accent-cyan)', fontSize: '11px' }}>
              ({selectedMachineType})
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {activeAssetStats ? (
              <>
                <span style={{ color: 'var(--text-secondary)' }}>
                  History: <strong>{activeAssetStats.total_inspections}</strong> tests
                </span>
                <span
                  style={{
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 700,
                    background: activeAssetStats.anomaly_count > 0 ? 'var(--danger-bg)' : 'var(--success-bg)',
                    color: activeAssetStats.anomaly_count > 0 ? 'var(--danger)' : 'var(--success)',
                    border: `1px solid ${activeAssetStats.anomaly_count > 0 ? 'var(--danger-border)' : 'var(--success-border)'}`,
                  }}
                >
                  {activeAssetStats.anomaly_count} Anomalies Recorded
                </span>
              </>
            ) : (
              <span style={{ color: 'var(--accent-purple)', fontSize: '11px', fontWeight: 600 }}>
                ✨ Fresh Asset Profile (No prior records)
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
