"use client";
import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";

// Approximate Lloyd-Max centroids for N(0,1) at 4 bits
const CENTROIDS = [
  -2.401, -1.844, -1.437, -1.099, -0.7980, -0.5224, -0.2582, 0.0,
  0.2582, 0.5224, 0.7980, 1.099, 1.437, 1.844, 2.401, 2.401,
];
// Simplified 16-entry set (symmetric, excluding duplicate last)
const C16 = [
  -2.40, -1.84, -1.44, -1.10, -0.80, -0.52, -0.26, 0.0,
  0.26, 0.52, 0.80, 1.10, 1.44, 1.84, 2.40,
];

function gaussian(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

export function GaussianQuantizationViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.3 });
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let start: number;
    const duration = 2000;
    const tick = (now: number) => {
      if (!start) start = now;
      const p = Math.min((now - start) / duration, 1);
      setProgress(1 - Math.pow(1 - p, 3)); // ease out
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
    const pad = { l: 40, r: 20, t: 20, b: 40 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    const xMin = -3.5, xMax = 3.5;
    const yMax = 0.45;

    const toX = (v: number) => pad.l + ((v - xMin) / (xMax - xMin)) * plotW;
    const toY = (v: number) => pad.t + (1 - v / yMax) * plotH;

    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#30363d";
    ctx.lineWidth = 0.5;
    for (let x = -3; x <= 3; x++) {
      ctx.beginPath();
      ctx.moveTo(toX(x), pad.t);
      ctx.lineTo(toX(x), H - pad.b);
      ctx.stroke();
    }

    // Axis
    ctx.strokeStyle = "#8b949e";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, H - pad.b);
    ctx.lineTo(W - pad.r, H - pad.b);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#8b949e";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    for (let x = -3; x <= 3; x++) {
      ctx.fillText(String(x), toX(x), H - pad.b + 16);
    }

    // Gaussian curve
    ctx.beginPath();
    const numVisible = Math.floor(progress * 300);
    for (let i = 0; i <= numVisible; i++) {
      const x = xMin + (i / 300) * (xMax - xMin);
      const y = gaussian(x);
      if (i === 0) ctx.moveTo(toX(x), toY(y));
      else ctx.lineTo(toX(x), toY(y));
    }
    ctx.strokeStyle = "#58a6ff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill under curve
    if (progress > 0.3) {
      const fillProgress = Math.min((progress - 0.3) / 0.4, 1);
      ctx.beginPath();
      ctx.moveTo(toX(xMin), toY(0));
      for (let i = 0; i <= 300; i++) {
        const x = xMin + (i / 300) * (xMax - xMin);
        ctx.lineTo(toX(x), toY(gaussian(x)));
      }
      ctx.lineTo(toX(xMax), toY(0));
      ctx.closePath();
      ctx.fillStyle = `rgba(88, 166, 255, ${0.06 * fillProgress})`;
      ctx.fill();
    }

    // Centroids
    if (progress > 0.5) {
      const centroidProgress = Math.min((progress - 0.5) / 0.4, 1);
      const numCentroids = Math.floor(centroidProgress * C16.length);

      for (let i = 0; i < numCentroids; i++) {
        const c = C16[i];
        const cx = toX(c);
        const cy = toY(gaussian(c));

        // Vertical line
        ctx.beginPath();
        ctx.moveTo(cx, toY(0));
        ctx.lineTo(cx, cy);
        ctx.strokeStyle = `rgba(126, 231, 135, ${0.6 * centroidProgress})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Dot at top
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#7ee787";
        ctx.fill();

        // Index label
        ctx.fillStyle = "#7ee787";
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(String(i), cx, toY(0) + 12);
      }

      // Decision boundaries
      if (centroidProgress > 0.5) {
        const bndProgress = (centroidProgress - 0.5) / 0.5;
        ctx.setLineDash([3, 3]);
        for (let i = 0; i < C16.length - 1; i++) {
          const mid = (C16[i] + C16[i + 1]) / 2;
          const mx = toX(mid);
          ctx.beginPath();
          ctx.moveTo(mx, pad.t);
          ctx.lineTo(mx, H - pad.b);
          ctx.strokeStyle = `rgba(210, 168, 255, ${0.3 * bndProgress})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        ctx.setLineDash([]);
      }
    }
  }, [progress]);

  return (
    <div ref={ref}>
      <canvas
        ref={canvasRef}
        width={700}
        height={300}
        className="w-full max-w-[700px] mx-auto"
        style={{ imageRendering: "auto" }}
      />
      <div className="flex items-center justify-center gap-6 mt-4 text-xs text-txt-2">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-accent inline-block rounded" /> Gaussian 𝒩(0,1)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-accent-green rounded-full inline-block" /> Centroids
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-accent-purple/50 inline-block rounded" style={{ borderTop: "1px dashed" }} /> Boundaries
        </span>
      </div>
    </div>
  );
}
