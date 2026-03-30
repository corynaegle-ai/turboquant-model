"use client";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";

interface Step {
  icon: string;
  label: string;
  detail: string;
  color: string;
}

export function PipelineFlow({ steps }: { steps: Step[] }) {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.2 });

  return (
    <div ref={ref} className="relative py-8">
      {/* Flow line */}
      <div className="absolute top-1/2 left-[5%] right-[5%] h-[3px] -translate-y-1/2 rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-gradient-to-r from-accent via-accent-green to-accent-purple"
          initial={{ scaleX: 0 }}
          animate={inView ? { scaleX: 1 } : {}}
          transition={{ duration: 1.5, ease: "easeInOut" }}
          style={{ transformOrigin: "left" }}
        />
      </div>

      {/* Flow dot */}
      {inView && (
        <motion.div
          className="absolute top-1/2 w-3 h-3 bg-accent rounded-full -translate-y-1/2 shadow-lg shadow-accent/50"
          animate={{ left: ["5%", "95%"], opacity: [0, 1, 1, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
      )}

      {/* Steps */}
      <div className="relative flex items-center justify-between gap-2 flex-wrap">
        {steps.map((step, i) => (
          <motion.div
            key={step.label}
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.15 * i, duration: 0.4 }}
            className="bg-bg-2 border border-border rounded-xl px-4 py-3 text-center min-w-[110px] relative z-10 hover:border-accent hover:shadow-lg hover:shadow-accent/10 hover:-translate-y-1 transition-all cursor-default"
            style={{ borderColor: `${step.color}40` }}
          >
            <div className="text-2xl mb-1">{step.icon}</div>
            <div className="text-xs font-semibold">{step.label}</div>
            <div className="text-[10px] text-txt-2 mt-0.5">{step.detail}</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
