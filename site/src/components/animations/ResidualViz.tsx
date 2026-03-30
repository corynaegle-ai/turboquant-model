"use client";
import { useEffect, useRef, useState } from "react";
import { useInView } from "react-intersection-observer";

function seededRandom(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

export function ResidualViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.3 });
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let start: number;
    const tick = (now: number) => {
      if (!start) start = now;
      const p = Math.min((now - start) / 4000, 1);
      setProgress(p);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [inView]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const N = 60;
    const rowH = H / 4;
    const pad = { l: 90, r: 20 };
    const plotW = W - pad.l - pad.r;

    ctx.clearRect(0, 0, W, H);

    // Generate signal
    const signal: number[] = [];
    for (let i = 0; i < N; i++) {
      signal.push((seededRandom(i * 7 + 3) - 0.5) * 2 + Math.sin(i * 0.3) * 0.5);
    }

    // Quantize (coarse)
    const quantized: number[] = signal.map((v) => {
      const level = Math.round(v * 3) / 3;
      return Math.max(-1, Math.min(1, level));
    });

    // Residual
    const residual: number[] = signal.map((v, i) => v - quantized[i]);

    // Quantize residual
    const residualQ: number[] = residual.map((v) => {
      const level = Math.round(v * 8) / 8;
      return Math.max(-0.5, Math.min(0.5, level));
    });

    // Reconstructed
    const reconstructed: number[] = quantized.map((v, i) => v + residualQ[i]);

    type RowData = { label: string; data: number[]; color: string; phase: [number, number]; maxAmp: number };
    const rows: RowData[] = [
      { label: "Original W", data: signal, color: "#58a6ff", phase: [0, 0.25], maxAmp: 1.2 },
      { label: "Pass 1 (Ŵ₁)", data: quantized, color: "#7ee787", phase: [0.25, 0.5], maxAmp: 1.2 },
      { label: "Residual (R)", data: residual, color: "#f85149", phase: [0.5, 0.75], maxAmp: 0.6 },
      { label: "Ŵ₁ + R̂ ≈ W", data: reconstructed, color: "#d2a8ff", phase: [0.75, 1.0], maxAmp: 1.2 },
    ];

    rows.forEach((row, ri) => {
      const y0 = ri * rowH;
      const localProgress = Math.max(0, Math.min(1, (progress - row.phase[0]) / (row.phase[1] - row.phase[0])));

      // Label
      ctx.fillStyle = row.color;
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "right";
      ctx.globalAlpha = Math.min(localProgress * 3, 1);
      ctx.fillText(row.label, pad.l - 10, y0 + rowH / 2 + 4);

      // Baseline
      ctx.strokeStyle = "#30363d";
      ctx.lineWidth = 0.5;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(pad.l, y0 + rowH / 2);
      ctx.lineTo(W - pad.r, y0 + rowH / 2);
      ctx.stroke();

      // Signal bars
      const numBars = Math.floor(localProgress * N);
      const barW = plotW / N - 1;

      for (let i = 0; i < numBars; i++) {
        const x = pad.l + (i / N) * plotW;
        const val = row.data[i];
        const barH = (val / row.maxAmp) * (rowH * 0.4);

        ctx.globalAlpha = 0.8;
        ctx.fillStyle = row.color;
        ctx.fillRect(x, y0 + rowH / 2 - Math.max(0, barH), barW, Math.abs(barH));
      }

      ctx.globalAlpha = 1;
    });
  }, [progress]);

  return (
    <div ref={ref}>
      <canvas
        ref={canvasRef}
        width={700}
        height={320}
        className="w-full max-w-[700px] mx-auto"
      />
      <div className="flex items-center justify-center gap-4 mt-3 text-xs text-txt-2 flex-wrap">
        <span className="flex items-center gap-1"><span className="w-3 h-2 bg-accent rounded-sm inline-block" /> Original</span>
        <span className="flex items-center gap-1"><span className="w-3 h-2 bg-accent-green rounded-sm inline-block" /> Pass 1</span>
        <span className="flex items-center gap-1"><span className="w-3 h-2 rounded-sm inline-block" style={{ background: "#f85149" }} /> Residual</span>
        <span className="flex items-center gap-1"><span className="w-3 h-2 bg-accent-purple rounded-sm inline-block" /> Reconstructed</span>
      </div>
    </div>
  );
}
