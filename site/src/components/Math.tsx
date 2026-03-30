"use client";
import { useEffect, useRef } from "react";
import katex from "katex";

interface MathProps {
  expr: string;
  display?: boolean;
  className?: string;
}

export function Math({ expr, display = false, className = "" }: MathProps) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (ref.current) {
      katex.render(expr, ref.current, {
        displayMode: display,
        throwOnError: false,
      });
    }
  }, [expr, display]);

  return <span ref={ref} className={className} />;
}
