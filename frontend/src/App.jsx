import React, { useState, useEffect } from 'react';
import { Activity, Radio, Cpu, RefreshCw, Info, ShieldCheck } from 'lucide-react';
import AudioUploader from './components/AudioUploader';
import Spectrogram from './components/Spectrogram';
import AnomalyGauge from './components/AnomalyGauge';
import MetricsCard from './components/MetricsCard';
import LLMAdvisorCard from './components/LLMAdvisorCard';
import AnomalyActionMenu from './components/AnomalyActionMenu';
import MachinePrognosticsCard from './components/MachinePrognosticsCard';
import { analyzeAudioFile, analyzePreset, fetchModelProfile, checkBackendHealth } from './services/api';

export default function App() {
  const [selectedMachine, setSelectedMachine] = useState('pump');
  const [selectedMachineId, setSelectedMachineId] = useState('p001');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Load initial model profile and check backend status
  useEffect(() => {
    async function init() {
      const isOnline = await checkBackendHealth();
      setBackendOnline(isOnline);
      const profile = await fetchModelProfile();
      setProfileData(profile);
    }
    init();
  }, []);

  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    setIsAnalyzing(true);
    try {
      const result = await analyzeAudioFile(file, false, selectedMachine, selectedMachineId);
      setAnalysisResult(result);
      setRefreshTrigger((prev) => prev + 1);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSampleSelect = async (sampleName, isAnomaly, machineOverride) => {
    const machine = machineOverride || selectedMachine;
    setSelectedFile({ name: sampleName });
    setIsAnalyzing(true);
    try {
      const result = await analyzePreset(sampleName, isAnomaly, machine, selectedMachineId);
      setAnalysisResult(result);
      setRefreshTrigger((prev) => prev + 1);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleMachineTypeChange = (machineType) => {
    setSelectedMachine(machineType);
    setSelectedFile(null);
    setAnalysisResult(null);
    const prefix = machineType === 'pump' ? 'p' : machineType === 'valve' ? 'v' : 'f';
    setSelectedMachineId(`${prefix}001`);
  };

  const handleMachineIdChange = (newMachineId) => {
    setSelectedMachineId(newMachineId);
  };

  const handleAnomalyAssigned = (targetMachineId) => {
    setSelectedMachineId(targetMachineId);
    setRefreshTrigger((prev) => prev + 1);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon">
            <Radio size={26} />
          </div>
          <div>
            <h1 className="brand-title">VISION</h1>
            <p className="brand-subtitle">TinyML Industrial Acoustic Anomaly Detector & Predictive Maintenance Hub</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div className="header-badge">
            <span className="pulse-dot" style={{ backgroundColor: backendOnline ? 'var(--success)' : 'var(--accent-cyan)' }} />
            {backendOnline ? 'BACKEND CONNECTED' : 'EDGE DEMO ENGINE'}
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Left Column: Audio Ingestion & Spectrogram */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Card: Audio Uploader with Machine Asset Selection */}
          <div className="glass-panel">
            <div className="card-header">
              <h2 className="card-title">
                <Radio size={18} color="var(--accent-cyan)" />
                1 · Audio Processor & Asset Selector
              </h2>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Inspector's Ears</span>
            </div>
            <div className="card-body">
              <AudioUploader
                selectedMachine={selectedMachine}
                onMachineChange={handleMachineTypeChange}
                selectedMachineId={selectedMachineId}
                onMachineIdChange={handleMachineIdChange}
                refreshTrigger={refreshTrigger}
                onFileSelected={handleFileSelect}
                onSampleSelected={handleSampleSelect}
                isAnalyzing={isAnalyzing}
                selectedFileName={selectedFile?.name}
              />
            </div>
          </div>

          {/* Card: Spectrogram Heatmap */}
          <div className="glass-panel">
            <div className="card-header">
              <h2 className="card-title">
                <Activity size={18} color="var(--accent-purple)" />
                Frequency Spectrum Analysis
              </h2>
              {isAnalyzing && (
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--accent-cyan)' }}>
                  <RefreshCw size={14} className="pulse-dot" /> Computing FFT...
                </span>
              )}
            </div>
            <div className="card-body">
              {analysisResult ? (
                <Spectrogram spectrogramMatrix={analysisResult.spectrogram} />
              ) : (
                <div style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--text-muted)' }}>
                  <Activity size={32} color="var(--accent-purple)" style={{ opacity: 0.35, marginBottom: '10px' }} />
                  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    No Audio Clip Loaded
                  </div>
                  <div style={{ fontSize: '12px', marginTop: '4px' }}>
                    Upload or select a sample above to compute 64-band Log-Mel frequencies
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: AI Brain, Decision & Anomaly Action Menu */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Card: AI Decision & Reconstruction Gauge */}
          <div className="glass-panel">
            <div className="card-header">
              <h2 className="card-title">
                <Cpu size={18} color="var(--accent-blue)" />
                2 · The AI Brain (PyTorch Autoencoder)
              </h2>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Reconstruction Engine</span>
            </div>
            <div className="card-body">
              {analysisResult ? (
                <>
                  <AnomalyGauge
                    score={analysisResult.score}
                    threshold={analysisResult.threshold}
                    isAnomaly={analysisResult.is_anomaly}
                    analysisResult={analysisResult}
                  />

                  {/* Interactive Anomaly Action Menu if Anomaly is detected */}
                  {analysisResult.is_anomaly && (
                    <AnomalyActionMenu
                      analysisResult={analysisResult}
                      selectedMachineType={selectedMachine}
                      currentMachineId={selectedMachineId}
                      onAnomalyAssigned={handleAnomalyAssigned}
                    />
                  )}
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '52px 20px', color: 'var(--text-muted)' }}>
                  <Cpu size={36} color="var(--accent-cyan)" style={{ opacity: 0.35, marginBottom: '12px' }} />
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Ready to Inspect {selectedMachine === 'pump' ? 'Pump' : selectedMachine === 'valve' ? 'Valve' : 'Fan'} ({selectedMachineId})
                  </div>
                  <div style={{ fontSize: '12px', marginTop: '6px', color: 'var(--text-muted)' }}>
                    Baseline calibrated at τ = {selectedMachine === 'pump' ? '0.0201' : selectedMachine === 'valve' ? '0.0155' : '0.0179'}. Awaiting audio input.
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Card: Digital Factory Inspector Explanation */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan)' }}>
              <Info size={16} /> How the Digital Inspector Decides:
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              The <strong>Autoencoder</strong> compressed this audio down to an 8×8 bottleneck and reconstructed the original sound. Normal machine hum reconstructs easily (low error). When damaged gears, bearing rattles, or fluid leaks occur, high reconstruction error triggers the <strong>🔴 ANOMALY</strong> alert, allowing you to log it to the machine's historical record.
            </p>
          </div>
        </div>
      </main>

      {/* Machine Asset Prognostics & Predictive Fault Analysis */}
      <MachinePrognosticsCard
        machineId={selectedMachineId}
        machineType={selectedMachine}
        refreshTrigger={refreshTrigger}
      />

      {/* Gemini LLM Anomaly Diagnostics & Maintenance Advisory */}
      {analysisResult && (
        <LLMAdvisorCard
          machineType={selectedMachine}
          analysisResult={analysisResult}
        />
      )}

      {/* Edge Performance & Profiler Section */}
      <section className="glass-panel">
        <div className="card-header">
          <h2 className="card-title">
            <ShieldCheck size={18} color="var(--success)" />
            3 & 4 · The Shrink Tool (TFLite INT8) & Profiler Metrics
          </h2>
          <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            Target: &lt; 256 KB Flash
          </span>
        </div>
        <div className="card-body" style={{ paddingTop: '8px' }}>
          <MetricsCard profile={profileData} />
        </div>
      </section>
    </div>
  );
}
