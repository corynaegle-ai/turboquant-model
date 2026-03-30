"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { ResidualViz } from "@/components/animations/ResidualViz";
import { Reveal } from "@/components/Reveal";

export default function ResidualPage() {
  return (
    <TechniqueLayout
      title="Residual Quantization"
      subtitle="Multi-pass quantization captures progressively finer detail. Each pass quantizes the error left by the previous pass."
      color="#ffa657"
      icon="🎯"
      prev={{ href: "/techniques/walsh-hadamard/", label: "Walsh-Hadamard" }}
      next={{ href: "/techniques/packing/", label: "4-bit Packing" }}
    >
      <Section title="Motivation">
        <p className="text-txt-2 leading-relaxed">
          Single-pass quantization at <Math expr="b" /> bits has a fundamental error floor —
          the Lloyd-Max distortion for <Math expr="\mathcal{N}(0,1)" /> at 4 bits. For LLMs,
          this translates to ~2 PPL degradation on Qwen3.5-0.8B. Residual quantization
          dramatically reduces this.
        </p>
      </Section>

      <Section title="The Idea">
        <div className="bg-bg-2 border border-border rounded-xl p-6 space-y-2 font-mono text-sm">
          <div><Math expr="R_0 = W" /></div>
          <div><Math expr="\hat{W}_k = \text{TQ}(R_{k-1}, b_k\text{ bits}, \text{seed}_k)" /></div>
          <div><Math expr="R_k = R_{k-1} - \hat{W}_k" /></div>
          <div className="pt-2 border-t border-border">
            <Math expr="W_{\text{approx}} = \sum_{k=1}^{n} \hat{W}_k" />
          </div>
        </div>
        <p className="text-txt-2 leading-relaxed mt-4">
          Each pass captures progressively finer detail. The residual after pass <Math expr="k" />{" "}
          has smaller magnitude, so even a coarse quantizer captures significant information.
        </p>
      </Section>

      <Section title="Animated: Two-Pass Residual">
        <p className="text-txt-2 mb-6 leading-relaxed">
          Watch the signal decompose: Pass 1 captures coarse structure, the residual contains
          the fine details, and Pass 2 captures most of them.
        </p>
        <div className="bg-bg-2 border border-border rounded-2xl p-6">
          <ResidualViz />
        </div>
      </Section>

      <Section title="Why 4+4 Beats Single-Pass 8-bit">
        <div className="grid sm:grid-cols-2 gap-4">
          <Reveal>
            <div className="bg-bg-3 border border-border rounded-xl p-5">
              <h4 className="text-accent-orange font-semibold text-sm mb-2">Single-pass 8-bit</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Allocates 256 levels uniformly across the whole dynamic range. Most levels wasted
                in low-density regions.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="bg-bg-3 border border-accent-green/30 rounded-xl p-5">
              <h4 className="text-accent-green font-semibold text-sm mb-2">4+4 residual ✨</h4>
              <p className="text-xs text-txt-2 leading-relaxed">
                Pass 1: 16 levels optimized for original distribution.
                Pass 2: 16 levels optimized for the residual distribution. Two-stage allocation
                is far more efficient.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>

      <Section title="Results">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent text-xs uppercase">Config</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Bits</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">PPL</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">KLD</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border"><td className="py-3 px-4 text-txt-2">Baseline bf16</td><td className="py-3 px-4">16</td><td className="py-3 px-4">14.29</td><td className="py-3 px-4">—</td></tr>
              <tr className="border-b border-border bg-accent-green/5"><td className="py-3 px-4 text-accent-green font-semibold">4+4 residual ✨</td><td className="py-3 px-4 text-accent-green font-semibold">8</td><td className="py-3 px-4 text-accent-green font-semibold">14.28</td><td className="py-3 px-4 text-accent-green font-semibold">0.0020</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4">4+2 residual</td><td className="py-3 px-4">6</td><td className="py-3 px-4">14.46</td><td className="py-3 px-4">0.0159</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4">3+2 residual</td><td className="py-3 px-4">5</td><td className="py-3 px-4">15.15</td><td className="py-3 px-4">0.0545</td></tr>
              <tr><td className="py-3 px-4">4-bit single</td><td className="py-3 px-4">4</td><td className="py-3 px-4">16.58</td><td className="py-3 px-4">0.1403</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">residual.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">residual_quantize_packed()</span>{" "}
            <span className="text-txt-2">(independent seeds)</span>
          </div>
          <div>
            <span className="text-accent-purple">residual.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">multi_residual_quantize_packed()</span>{" "}
            <span className="text-txt-2">(shared seed)</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
