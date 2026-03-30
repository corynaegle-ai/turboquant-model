"use client";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";

const N = 8;
const STAGES = 3;

// Butterfly connections for 8-point FWHT
function getConnections() {
  const conns: Array<{ fromStage: number; fromIdx: number; toStage: number; toIdx: number; cross: boolean }> = [];
  for (let s = 0; s < STAGES; s++) {
    const half = 1 << s;
    const block = 1 << (s + 1);
    for (let b = 0; b < N; b += block) {
      for (let k = 0; k < half; k++) {
        const i = b + k;
        const j = b + k + half;
        // straight
        conns.push({ fromStage: s, fromIdx: i, toStage: s + 1, toIdx: i, cross: false });
        conns.push({ fromStage: s, fromIdx: j, toStage: s + 1, toIdx: j, cross: false });
        // cross
        conns.push({ fromStage: s, fromIdx: i, toStage: s + 1, toIdx: j, cross: true });
        conns.push({ fromStage: s, fromIdx: j, toStage: s + 1, toIdx: i, cross: true });
      }
    }
  }
  return conns;
}

export function ButterflyDiagram() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.3 });
  const connections = getConnections();

  const xGap = 90;
  const yGap = 32;
  const xOff = 40;
  const yOff = 20;
  const W = xOff * 2 + STAGES * xGap;
  const H = yOff * 2 + (N - 1) * yGap;

  const nodeX = (stage: number) => xOff + stage * xGap;
  const nodeY = (idx: number) => yOff + idx * yGap;

  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[600px] mx-auto">
        {/* Connections */}
        {connections.map((c, i) => {
          const x1 = nodeX(c.fromStage);
          const y1 = nodeY(c.fromIdx);
          const x2 = nodeX(c.toStage);
          const y2 = nodeY(c.toIdx);
          return (
            <motion.line
              key={`conn-${i}`}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={c.cross ? "#d2a8ff" : "#58a6ff"}
              strokeWidth={1}
              opacity={0}
              animate={inView ? { opacity: c.cross ? 0.35 : 0.5 } : {}}
              transition={{ delay: c.fromStage * 0.3 + i * 0.01, duration: 0.4 }}
            />
          );
        })}

        {/* Nodes */}
        {Array.from({ length: STAGES + 1 }, (_, s) =>
          Array.from({ length: N }, (_, i) => (
            <motion.circle
              key={`node-${s}-${i}`}
              cx={nodeX(s)}
              cy={nodeY(i)}
              r={5}
              fill={s === 0 ? "#58a6ff" : s === STAGES ? "#7ee787" : "#d2a8ff"}
              initial={{ scale: 0, opacity: 0 }}
              animate={inView ? { scale: 1, opacity: 1 } : {}}
              transition={{ delay: s * 0.25 + i * 0.03, duration: 0.3, type: "spring" }}
            />
          ))
        )}

        {/* Stage labels */}
        {["input", "stage 0", "stage 1", "output"].map((label, s) => (
          <motion.text
            key={label}
            x={nodeX(s)}
            y={H - 2}
            textAnchor="middle"
            fill={s === 0 ? "#58a6ff" : s === 3 ? "#7ee787" : "#8b949e"}
            fontSize={10}
            fontFamily="inherit"
            initial={{ opacity: 0 }}
            animate={inView ? { opacity: 1 } : {}}
            transition={{ delay: s * 0.25 + 0.5, duration: 0.4 }}
          >
            {label}
          </motion.text>
        ))}
      </svg>

      <p className="text-center text-xs text-txt-2 mt-3">
        8-point Fast Walsh-Hadamard Transform butterfly pattern (3 stages, O(n log n) operations)
      </p>
    </div>
  );
}
