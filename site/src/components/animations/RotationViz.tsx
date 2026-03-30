"use client";
import { useRef, useEffect, useState, useCallback } from "react";
import { useInView } from "react-intersection-observer";

function seededRandom(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

export function RotationViz() {
  const beforeRef = useRef<HTMLCanvasElement>(null);
  const afterRef = useRef<HTMLCanvasElement>(null);
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.3 });
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let start: number;
    const tick = (now: number) => {
      if (!start) start = now;
      const p = Math.min((now - start) / 2500, 1);
      setProgress(1 - Math.pow(1 - p, 3));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [inView]);

  const drawScatter = useCallback(
    (canvas: HTMLCanvasElement | null, rotated: boolean) => {
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      // Axes
      ctx.strokeStyle = "#30363d";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
      ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
      ctx.stroke();

      const numPoints = Math.floor(progress * 200);
      const scale = W / 7;

      for (let i = 0; i < numPoints; i++) {
        let x: number, y: number;
        if (!rotated) {
          // Correlated: elongated ellipse
          const u = seededRandom(i * 2) * 6 - 3;
          const v = seededRandom(i * 2 + 1) * 2 - 1;
          x = u * 0.8 + v * 0.6;
          y = u * 0.6 - v * 0.3;
        } else {
          // After rotation: roughly circular Gaussian
          const u1 = seededRandom(i * 3 + 100);
          const u2 = seededRandom(i * 3 + 101);
          const r = Math.sqrt(-2 * Math.log(Math.max(u1, 0.001)));
          x = r * Math.cos(2 * Math.PI * u2);
          y = r * Math.sin(2 * Math.PI * u2);
        }

        const px = W / 2 + x * scale;
        const py = H / 2 + y * scale;

        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = rotated
          ? `rgba(126, 231, 135, ${0.6 + seededRandom(i) * 0.3})`
          : `rgba(88, 166, 255, ${0.4 + seededRandom(i) * 0.4})`;
        ctx.fill();
      }

      // Label
      ctx.fillStyle = "#8b949e";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(rotated ? "After rotation (≈ i.i.d. Gaussian)" : "Original weights (correlated)", W / 2, H - 8);
    },
    [progress]
  );

  useEffect(() => {
    drawScatter(beforeRef.current, false);
    drawScatter(afterRef.current, true);
  }, [drawScatter]);

  return (
    <div ref={ref} className="flex flex-col md:flex-row items-center gap-6">
      <div className="flex-1 w-full">
        <canvas ref={beforeRef} width={300} height={300} className="w-full max-w-[300px] mx-auto" />
      </div>
      <div className="text-3xl text-accent animate-pulse">→</div>
      <div className="flex-1 w-full">
        <canvas ref={afterRef} width={300} height={300} className="w-full max-w-[300px] mx-auto" />
      </div>
    </div>
  );
}
