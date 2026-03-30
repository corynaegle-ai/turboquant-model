"use client";
import { motion } from "framer-motion";

const naiveSteps = [
  { label: "📦 Unpack uint8 → int64", badge: "Global Memory", color: "red" },
  { label: "📖 Codebook lookup → float32", badge: "Global Memory", color: "red" },
  { label: "✖️ Matrix multiply", badge: "Global Memory", color: "red" },
  { label: "⚖️ Rescale", badge: "Global Memory", color: "red" },
];

const fusedSteps = [
  { label: "📦 Load packed uint8", badge: "Registers", color: "green" },
  { label: "🔓 Unpack nibbles (bitwise)", badge: "Registers", color: "green" },
  { label: "📖 Codebook (64B in L1)", badge: "Shared Mem", color: "green" },
  { label: "✖️ Tensor Core MMA + Rescale", badge: "Registers", color: "green" },
  { label: "💾 Store final result", badge: "1× Global Write", color: "green" },
];

export function KernelFusionViz() {
  return (
    <div className="grid md:grid-cols-2 gap-6">
      {/* Naive */}
      <div className="bg-bg-3 border border-border rounded-xl p-5">
        <h4 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-red-500" />
          Naive Pipeline (4 kernel launches)
        </h4>
        <div className="space-y-2">
          {naiveSteps.map((s, i) => (
            <motion.div
              key={`naive-${i}`}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12, duration: 0.3 }}
            >
              <div className="flex items-center gap-3 bg-red-500/5 border border-red-500/15 rounded-lg px-3 py-2">
                <span className="text-xs">{s.label}</span>
                <span className="ml-auto text-[10px] px-2 py-0.5 rounded bg-red-500/10 text-red-400 whitespace-nowrap">
                  {s.badge}
                </span>
              </div>
              {i < naiveSteps.length - 1 && (
                <div className="text-center text-[10px] text-txt-2 py-0.5">↓ write + read</div>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Fused */}
      <div className="bg-bg-3 border border-accent-green/20 rounded-xl p-5">
        <h4 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-accent-green" />
          Fused Kernel (1 kernel launch)
        </h4>
        <div className="space-y-2">
          {fusedSteps.map((s, i) => (
            <motion.div
              key={`fused-${i}`}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 + 0.3, duration: 0.3 }}
            >
              <div className="flex items-center gap-3 bg-accent-green/5 border border-accent-green/15 rounded-lg px-3 py-2">
                <span className="text-xs">{s.label}</span>
                <span className="ml-auto text-[10px] px-2 py-0.5 rounded bg-accent-green/10 text-accent-green whitespace-nowrap">
                  {s.badge}
                </span>
              </div>
              {i < fusedSteps.length - 1 && (
                <div className="text-center text-[10px] text-txt-2 py-0.5">↓ in-register</div>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
