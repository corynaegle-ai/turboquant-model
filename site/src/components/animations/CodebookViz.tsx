"use client";
import { motion } from "framer-motion";

const entries = [
  { idx: 0, val: "-2.40" }, { idx: 1, val: "-1.84" },
  { idx: 2, val: "-1.44" }, { idx: 3, val: "-1.10" },
  { idx: 4, val: "-0.80" }, { idx: 5, val: "-0.52" },
  { idx: 6, val: "-0.26" }, { idx: 7, val: "0.00" },
  { idx: 8, val: "+0.26" }, { idx: 9, val: "+0.52" },
  { idx: 10, val: "+0.80" }, { idx: 11, val: "+1.10" },
  { idx: 12, val: "+1.44" }, { idx: 13, val: "+1.84" },
  { idx: 14, val: "+2.40" }, { idx: 15, val: "+2.40" },
];

export function CodebookViz() {
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {entries.map((e, i) => {
        const t = Math.abs(parseFloat(e.val)) / 2.4;
        const hue = 210 + t * 80;
        return (
          <motion.div
            key={e.idx}
            initial={{ scale: 0.5, opacity: 0 }}
            whileInView={{ scale: 1, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.04, duration: 0.3, ease: "backOut" }}
            whileHover={{ scale: 1.15, zIndex: 10 }}
            className="w-14 h-14 rounded-lg flex flex-col items-center justify-center font-mono cursor-default transition-shadow"
            style={{
              background: `hsla(${hue}, 60%, 50%, 0.15)`,
              border: `1px solid hsla(${hue}, 60%, 50%, 0.3)`,
            }}
          >
            <span className="text-xs font-bold">{e.idx}</span>
            <span className="text-[10px] text-txt-2">{e.val}</span>
          </motion.div>
        );
      })}
    </div>
  );
}
