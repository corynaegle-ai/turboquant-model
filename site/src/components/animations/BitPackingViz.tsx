"use client";
import { motion } from "framer-motion";
import { useState } from "react";

const IDX_A = [0, 1, 0, 1]; // index 5 in binary
const IDX_B = [1, 0, 1, 1]; // index 11 in binary
const PACKED = [1, 0, 1, 1, 0, 1, 0, 1]; // hi:B lo:A

export function BitPackingViz() {
  const [step, setStep] = useState<"separate" | "packing" | "packed">("separate");

  return (
    <div className="space-y-8">
      {/* Controls */}
      <div className="flex justify-center gap-3">
        {(["separate", "packing", "packed"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setStep(s)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              step === s
                ? "bg-accent text-bg"
                : "bg-bg-3 border border-border text-txt-2 hover:border-accent"
            }`}
          >
            {s === "separate" ? "Two Indices" : s === "packing" ? "Packing..." : "Packed Byte"}
          </button>
        ))}
      </div>

      <div className="flex flex-col items-center gap-6">
        {/* Separate indices */}
        {(step === "separate" || step === "packing") && (
          <motion.div
            className="flex items-center gap-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: step === "packing" ? 0.5 : 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="text-center">
              <div className="text-xs text-txt-2 mb-2">idx[k] = 5</div>
              <div className="flex gap-1">
                {IDX_A.map((b, i) => (
                  <motion.div
                    key={`a-${i}`}
                    className="w-9 h-10 flex items-center justify-center rounded font-mono text-sm font-bold bg-accent/20 text-accent border border-accent/30"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: i * 0.08, type: "spring" }}
                  >
                    {b}
                  </motion.div>
                ))}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-txt-2 mb-2">idx[k+1] = 11</div>
              <div className="flex gap-1">
                {IDX_B.map((b, i) => (
                  <motion.div
                    key={`b-${i}`}
                    className="w-9 h-10 flex items-center justify-center rounded font-mono text-sm font-bold bg-accent-purple/20 text-accent-purple border border-accent-purple/30"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: (i + 4) * 0.08, type: "spring" }}
                  >
                    {b}
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* Arrow */}
        {step === "packing" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-2xl text-accent animate-pulse"
          >
            ↓ pack: lo | (hi &lt;&lt; 4)
          </motion.div>
        )}

        {/* Packed byte */}
        {(step === "packed" || step === "packing") && (
          <motion.div
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: step === "packed" ? 1 : 0.5, y: 0 }}
            transition={{ duration: 0.4, delay: step === "packed" ? 0 : 0.3 }}
          >
            <div className="text-xs text-txt-2 mb-2">packed uint8 = 0xB5 (181)</div>
            <div className="flex gap-0.5">
              {PACKED.map((b, i) => (
                <motion.div
                  key={`p-${i}`}
                  className={`w-9 h-10 flex items-center justify-center rounded font-mono text-sm font-bold ${
                    i < 4
                      ? "bg-accent-purple/20 text-accent-purple border border-accent-purple/30"
                      : "bg-accent/20 text-accent border border-accent/30"
                  }`}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: i * 0.06, type: "spring" }}
                >
                  {b}
                </motion.div>
              ))}
            </div>
            <div className="flex mt-1">
              <div className="flex-1 text-[10px] text-accent-purple">hi nibble (idx[k+1])</div>
              <div className="flex-1 text-[10px] text-accent">lo nibble (idx[k])</div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
