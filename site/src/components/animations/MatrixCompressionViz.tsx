"use client";
import { useEffect, useRef, useState } from "react";
import { useInView } from "react-intersection-observer";

function seededRandom(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

export function MatrixCompressionViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.3 });
  const [stage, setStage] = useState(0);

  useEffect(() => {
    if (!inView) return;
    const interval = setInterval(() => {
      setStage((s) => (s < 3 ? s + 1 : s));
    }, 1200);
    return () => clearInterval(interval);
  }, [inView]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const rows = 10;
    const cols = 10;
    const cellW = 24;
    const cellH = 24;
    const gap = 2;
    const labels = ["W (bf16)", "Y (rotated)", "idx (4-bit)", "packed (uint8)"];
    const colsPerStage = [cols, cols, cols, cols / 2];

    const totalW = colsPerStage[stage] * (cellW + gap);
    const totalH = rows * (cellH + gap);

    const offX = (W - totalW) / 2;
    const offY = (H - totalH - 20) / 2;

    const currentCols = colsPerStage[stage];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < currentCols; c++) {
        const x = offX + c * (cellW + gap);
        const y = offY + r * (cellH + gap);

        let color: string;
        const v = seededRandom(r * 17 + c * 31 + stage * 7);

        if (stage === 0) {
          // bf16 - varied blue/orange
          color = v > 0.5
            ? `rgba(88, 166, 255, ${v * 0.7 + 0.2})`
            : `rgba(255, 166, 87, ${v * 0.7 + 0.2})`;
        } else if (stage === 1) {
          // Rotated - uniform green
          color = `rgba(126, 231, 135, ${v * 0.5 + 0.2})`;
        } else if (stage === 2) {
          // Quantized - discrete hues
          const idx = Math.floor(v * 16);
          const hue = (idx / 16) * 280 + 200;
          color = `hsla(${hue % 360}, 65%, 55%, 0.7)`;
        } else {
          // Packed - purple
          color = `rgba(210, 168, 255, ${v * 0.6 + 0.2})`;
        }

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x, y, cellW, cellH, 3);
        ctx.fill();
      }
    }

    // Label
    ctx.fillStyle = "#8b949e";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(labels[stage], W / 2, offY + totalH + 24);

    // Stage indicator
    for (let i = 0; i < 4; i++) {
      ctx.beginPath();
      ctx.arc(W / 2 - 30 + i * 20, offY + totalH + 44, 4, 0, Math.PI * 2);
      ctx.fillStyle = i === stage ? "#58a6ff" : "#30363d";
      ctx.fill();
    }
  }, [stage, inView]);

  return (
    <div ref={ref} className="text-center">
      <canvas ref={canvasRef} width={400} height={350} className="mx-auto max-w-full" />
      <div className="flex justify-center gap-2 mt-3">
        {[0, 1, 2, 3].map((s) => (
          <button
            key={s}
            onClick={() => setStage(s)}
            className={`px-3 py-1 rounded text-xs transition-colors ${
              stage === s
                ? "bg-accent text-bg font-semibold"
                : "bg-bg-3 border border-border text-txt-2 hover:border-accent"
            }`}
          >
            {["bf16", "Rotated", "4-bit", "Packed"][s]}
          </button>
        ))}
      </div>
    </div>
  );
}
