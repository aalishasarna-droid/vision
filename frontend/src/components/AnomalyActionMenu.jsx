import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Save, Plus, ArrowRight, ShieldAlert, Cpu } from 'lucide-react';
import { reassignAnomalyRecord, fetchRegisteredMachines } from '../services/api';

export default function AnomalyActionMenu({
  analysisResult,
  selectedMachineType,
  currentMachineId,
  onAnomalyAssigned,
}) {
  const [mode, setMode] = useState('existing'); // 'existing' or 'new'
  const [existingId, setExistingId] = useState(currentMachineId || 'p001');
  const [newId, setNewId] = useState('');
  const [notes, setNotes] = useState('');
  const [machines, setMachines] = useState([]);
  const [isSaving, setIsSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [savedMachineId, setSavedMachineId] = useState('');

  // Load registered machines
  useEffect(() => {
    async function load() {
      const list = await fetchRegisteredMachines(selectedMachineType);
      setMachines(list);
      // Auto-suggest next new ID
      const prefix = selectedMachineType === 'pump' ? 'p' : selectedMachineType === 'valve' ? 'v' : 'f';
      setNewId(`${prefix}00${list.length + 1}`);
    }
    load();
  }, [selectedMachineType, analysisResult]);

  useEffect(() => {
    setExistingId(currentMachineId || 'p001');
    setSavedSuccess(false);
  }, [analysisResult, currentMachineId]);

  if (!analysisResult || !analysisResult.is_anomaly) {
    return null;
  }

  const recordId = analysisResult.detection_record?.recordId || analysisResult.recordId;

  const handleSave = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    const targetMachineId = mode === 'existing' ? existingId : newId.trim();

    try {
      if (recordId) {
        await reassignAnomalyRecord({
          recordId,
          machineId: targetMachineId,
          machineType: selectedMachineType,
          notes: notes.trim(),
        });
      }

      setSavedMachineId(targetMachineId);
      setSavedSuccess(true);
      if (onAnomalyAssigned) {
        onAnomalyAssigned(targetMachineId);
      }
    } catch (err) {
      console.error('Failed to assign anomaly:', err);
      alert('Error updating machine record: ' + err.message);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div
      className="anomaly-action-menu-card glass-panel"
      style={{
        border: '1px solid var(--danger-border)',
        background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(15, 23, 42, 0.95) 100%)',
        padding: '20px',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--shadow-glow-danger)',
        marginTop: '20px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'var(--danger-bg)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <AlertTriangle size={18} color="var(--danger)" />
          </div>
          <div>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
              Anomaly Detected — Factory Maintenance Action Required
            </h3>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>
              Acoustic error threshold exceeded (Score: {analysisResult.score?.toFixed(5)}). Choose how to store this fault event:
            </p>
          </div>
        </div>
      </div>

      {savedSuccess ? (
        <div
          style={{
            padding: '16px',
            borderRadius: '8px',
            background: 'var(--success-bg)',
            border: '1px solid var(--success-border)',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}
        >
          <CheckCircle size={22} color="var(--success)" />
          <div>
            <div style={{ fontWeight: 700, fontSize: '13px', color: 'var(--success)' }}>
              ✓ Anomaly Successfully Logged to Machine Record: {savedMachineId}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Historical recurrence frequency and predictive failure prognosis updated below.
            </div>
          </div>
        </div>
      ) : (
        <form onSubmit={handleSave}>
          {/* Mode Switcher: Existing vs New */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '10px',
              marginBottom: '16px',
            }}
          >
            <button
              type="button"
              onClick={() => setMode('existing')}
              style={{
                padding: '12px 14px',
                borderRadius: '8px',
                background: mode === 'existing' ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                border: mode === 'existing' ? '1px solid var(--accent-cyan)' : '1px solid var(--border-color)',
                color: mode === 'existing' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                fontWeight: 600,
                fontSize: '13px',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <Cpu size={16} />
              <div>
                <div>Log to Existing Machine ID</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400 }}>
                  Append to past fault history
                </div>
              </div>
            </button>

            <button
              type="button"
              onClick={() => setMode('new')}
              style={{
                padding: '12px 14px',
                borderRadius: '8px',
                background: mode === 'new' ? 'rgba(129, 140, 248, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                border: mode === 'new' ? '1px solid var(--accent-purple)' : '1px solid var(--border-color)',
                color: mode === 'new' ? 'var(--accent-purple)' : 'var(--text-secondary)',
                fontWeight: 600,
                fontSize: '13px',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <Plus size={16} />
              <div>
                <div>Register New Machine ID</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400 }}>
                  Initialize new asset history
                </div>
              </div>
            </button>
          </div>

          {/* Input based on mode */}
          {mode === 'existing' ? (
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Select Existing Machine ID:
              </label>
              <select
                value={existingId}
                onChange={(e) => setExistingId(e.target.value)}
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
                }}
              >
                {machines.map((m) => (
                  <option key={m.machineId} value={m.machineId}>
                    {m.machineId} — ({m.anomaly_count} past anomalies) • {m.latest_status.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div style={{ marginBottom: '14px' }}>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Enter New Machine ID / Serial:
              </label>
              <input
                type="text"
                value={newId}
                onChange={(e) => setNewId(e.target.value)}
                placeholder="e.g. p007, PUMP-EAST-02"
                required
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  background: 'rgba(15, 23, 42, 0.85)',
                  border: '1px solid var(--accent-purple)',
                  borderRadius: '8px',
                  color: 'var(--accent-purple)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '13px',
                  fontWeight: 600,
                }}
              />
            </div>
          )}

          {/* Maintenance Observation Notes */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
              Maintenance Engineer Shift Notes (Optional):
            </label>
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g. Suction cavitation rattle heard during high throughput load"
              style={{
                width: '100%',
                padding: '9px 12px',
                background: 'rgba(15, 23, 42, 0.7)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '12px',
              }}
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isSaving}
            className="btn btn-primary"
            style={{
              width: '100%',
              padding: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              background: 'linear-gradient(135deg, var(--danger) 0%, #b91c1c 100%)',
              border: 'none',
              fontWeight: 700,
              fontSize: '13px',
              cursor: isSaving ? 'not-allowed' : 'pointer',
            }}
          >
            <Save size={16} />
            {isSaving ? 'Logging Anomaly to Machine Record...' : `Save Anomaly to ${mode === 'existing' ? existingId : newId} Record & Run Prognostics`}
          </button>
        </form>
      )}
    </div>
  );
}
