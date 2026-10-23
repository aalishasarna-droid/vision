import React, { useState, useRef, useEffect } from 'react';
import { Upload, AlertTriangle, CheckCircle, Cpu, Radio } from 'lucide-react';
import MachineAssetSelector from './MachineAssetSelector';

const MACHINE_OPTIONS = [
  {
    id: 'pump',
    name: 'Pump',
    icon: '🚰',
    thresholdText: 'τ = 0.0201',
    samples: [
      { label: 'Normal Pump (ID 00)', file: 'normal_id_00_00000000.wav', isAnomaly: false },
      { label: 'Normal Pump (ID 02)', file: 'normal_id_02_00000000.wav', isAnomaly: false },
      { label: 'Anomaly Pump (ID 00)', file: 'anomaly_id_00_00000000.wav', isAnomaly: true },
      { label: 'Anomaly Pump (ID 02)', file: 'anomaly_id_02_00000000.wav', isAnomaly: true },
    ],
  },
  {
    id: 'valve',
    name: 'Valve',
    icon: '⚙️',
    thresholdText: 'τ = 0.0155',
    samples: [
      { label: 'Normal Valve (ID 00)', file: 'normal_id_00_00000000.wav', isAnomaly: false },
      { label: 'Normal Valve (ID 02)', file: 'normal_id_02_00000000.wav', isAnomaly: false },
      { label: 'Anomaly Valve (ID 00)', file: 'anomaly_id_00_00000000.wav', isAnomaly: true },
      { label: 'Anomaly Valve (ID 02)', file: 'anomaly_id_02_00000000.wav', isAnomaly: true },
    ],
  },
  {
    id: 'fan',
    name: 'Fan',
    icon: '🌀',
    thresholdText: 'τ = 0.0179',
    samples: [
      { label: 'Normal Fan (ID 00)', file: 'normal_id_00_00000000.wav', isAnomaly: false },
      { label: 'Normal Fan (ID 02)', file: 'normal_id_02_00000000.wav', isAnomaly: false },
      { label: 'Anomaly Fan (ID 00)', file: 'anomaly_id_00_00000000.wav', isAnomaly: true },
      { label: 'Anomaly Fan (ID 02)', file: 'anomaly_id_02_00000000.wav', isAnomaly: true },
    ],
  },
];

export default function AudioUploader({
  onFileSelected,
  onSampleSelected,
  selectedMachine = 'pump',
  onMachineChange,
  selectedMachineId = 'p001',
  onMachineIdChange,
  refreshTrigger = 0,
  isAnalyzing,
  selectedFileName,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [selectedMachine, selectedMachineId]);

  const activeMachine = MACHINE_OPTIONS.find((m) => m.id === selectedMachine) || MACHINE_OPTIONS[0];

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file && (file.type.includes('audio') || file.name.endsWith('.wav') || file.name.endsWith('.mp3'))) {
      onFileSelected(file);
    } else {
      alert('Please select a valid .wav or audio file.');
    }
  };

  return (
    <div>
      {/* Step 1 & 1B: Machine Asset Type & Machine ID Selection */}
      <MachineAssetSelector
        selectedMachineType={selectedMachine}
        onMachineTypeChange={onMachineChange}
        selectedMachineId={selectedMachineId}
        onMachineIdChange={onMachineIdChange}
        refreshTrigger={refreshTrigger}
      />

      {/* Step 2: Upload Audio Clip */}
      <div style={{ marginTop: '20px', marginBottom: '16px' }}>
        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Upload size={13} color="var(--accent-purple)" />
          Step 2 · Audio Inspection for {activeMachine.name} Asset ({selectedMachineId})
        </div>

        <div
          className={`dropzone ${isDragging ? 'active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            accept=".wav,.mp3,.flac"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div className="upload-icon">
            <Upload size={24} />
          </div>
          <h4 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '6px' }}>
            {selectedFileName ? selectedFileName : `Drop audio clip (.wav) for ${selectedMachineId} here`}
          </h4>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Analyzes Log-Mel reconstruction error against calibrated {activeMachine.name} threshold ({activeMachine.thresholdText})
          </p>
        </div>
      </div>

      {/* Preset Real Factory Samples for Chosen Machine */}
      <div className="sample-selector">
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          Or test with real MIMII {activeMachine.name} recordings:
        </span>
        <div className="sample-buttons">
          {activeMachine.samples.map((s, idx) => (
            <button
              key={idx}
              type="button"
              className="btn-sample"
              disabled={isAnalyzing}
              onClick={() => onSampleSelected(s.file, s.isAnomaly, activeMachine.id)}
            >
              {s.isAnomaly ? (
                <AlertTriangle size={14} color="var(--danger)" />
              ) : (
                <CheckCircle size={14} color="var(--success)" />
              )}
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
