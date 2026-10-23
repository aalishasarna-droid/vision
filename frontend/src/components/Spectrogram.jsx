import React, { useEffect, useRef } from 'react';

export default function Spectrogram({ spectrogramMatrix, title = 'Log-Mel Spectrogram (64 Mel Bins)' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !spectrogramMatrix || spectrogramMatrix.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const height = spectrogramMatrix.length;      // 64 bins
    const width = spectrogramMatrix[0].length;    // 64 frames

    // Set internal resolution matching data
    canvas.width = width;
    canvas.height = height;

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    // Colormap function (Magma / Inferno-inspired industrial gradient)
    const getColor = (normalizedVal) => {
      // normalizedVal in [0, 1]
      const v = Math.max(0, Math.min(1, (normalizedVal + 1) / 2)); // map [-1, 1] to [0, 1]
      
      let r, g, b;
      if (v < 0.25) {
        // Deep purple to navy
        r = Math.floor(v * 4 * 60);
        g = Math.floor(v * 4 * 10);
        b = Math.floor(70 + v * 4 * 120);
      } else if (v < 0.6) {
        // Magenta / Hot Pink
        const t = (v - 0.25) / 0.35;
        r = Math.floor(60 + t * 180);
        g = Math.floor(10 + t * 40);
        b = Math.floor(190 - t * 60);
      } else if (v < 0.85) {
        // Vivid Orange
        const t = (v - 0.6) / 0.25;
        r = Math.floor(240 + t * 15);
        g = Math.floor(50 + t * 130);
        b = Math.floor(130 - t * 110);
      } else {
        // Bright Yellow/White
        const t = (v - 0.85) / 0.15;
        r = 255;
        g = Math.floor(180 + t * 75);
        b = Math.floor(20 + t * 200);
      }

      return [r, g, b];
    };

    // Draw row by row (inverted so low frequencies are at bottom)
    for (let row = 0; row < height; row++) {
      const sourceRow = height - 1 - row; // flip Y for frequency axis
      for (let col = 0; col < width; col++) {
        const val = spectrogramMatrix[sourceRow][col];
        const [r, g, b] = getColor(val);
        const pixelIndex = (row * width + col) * 4;

        data[pixelIndex] = r;
        data[pixelIndex + 1] = g;
        data[pixelIndex + 2] = b;
        data[pixelIndex + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [spectrogramMatrix]);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>{title}</span>
        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
          Librosa 64-Mel × 64-Frame
        </span>
      </div>

      <div className="spectrogram-container">
        <canvas ref={canvasRef} className="spectrogram-canvas" />
        <div className="spectrogram-legend">
          <span>8000 Hz (Nyquist)</span>
          <span>Time: 0.0s → 2.0s</span>
          <span>0 Hz (DC)</span>
        </div>
      </div>
    </div>
  );
}
