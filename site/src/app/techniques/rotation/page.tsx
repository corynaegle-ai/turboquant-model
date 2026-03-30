"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { RotationViz } from "@/components/animations/RotationViz";
import { Reveal } from "@/components/Reveal";

export default function RotationPage() {
  return (
    <TechniqueLayout
      title="Random Rotation for Decorrelation"
      subtitle="Multiplying by a random orthogonal matrix spreads information uniformly across all coordinates — making each one approximately Gaussian."
      color="#7ee787"
      icon="🔄"
      prev={{ href: "/techniques/lloyd-max/", label: "Lloyd-Max" }}
      next={{ href: "/techniques/walsh-hadamard/", label: "Walsh-Hadamard" }}
    >
      <Section title="The Problem with Correlated Weights">
        <p className="text-txt-2 leading-relaxed">
          Neural network weight matrices have structured correlations — some coordinates carry
          much more information than others. Scalar quantization applied directly to correlated
          weights wastes bits on low-variance coordinates and under-quantizes high-variance ones.
        </p>
      </Section>

      <Section title="Solution: Random Orthogonal Rotation">
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math expr="Y = W_{\text{norm}} \cdot \Pi^T" display />
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            { title: "Norm-preserving", desc: "‖Y‖₂ = ‖W‖₂ = 1 — orthogonal matrices preserve norms" },
            { title: "Decorrelating", desc: "Coordinates of Y become approximately independent" },
            { title: "Gaussianizing", desc: "By CLT on high-dimensional unit vectors, each coordinate ≈ 𝒩(0, 1/d)" },
            { title: "Invertible", desc: "W = Y · Π — the inverse is just the transpose" },
          ].map((p, i) => (
            <Reveal key={p.title} delay={0.1 * i}>
              <div className="bg-bg-3 border border-border rounded-xl p-4">
                <h4 className="font-semibold text-accent-green text-sm mb-1">{p.title}</h4>
                <p className="text-xs text-txt-2">{p.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section title="Interactive: Before & After Rotation">
        <p className="text-txt-2 leading-relaxed mb-6">
          Watch how rotation transforms a structured weight distribution into a uniform
          Gaussian-like distribution where every coordinate has equal variance.
        </p>
        <div className="bg-bg-2 border border-border rounded-2xl p-6">
          <RotationViz />
        </div>
      </Section>

      <Section title="Haar-Distributed Random Orthogonal Matrix (QR Method)">
        <p className="text-txt-2 leading-relaxed mb-4">
          The &quot;gold standard&quot; for random rotations. Drawn from the Haar measure on O(d)
          — the unique distribution invariant under left/right multiplication by any orthogonal
          matrix.
        </p>
        <div className="space-y-3">
          {[
            { step: "1", text: "Draw A ∈ ℝ^{d×d} with i.i.d. 𝒩(0,1) entries" },
            { step: "2", text: "Compute QR decomposition: A = QR" },
            { step: "3", text: "Adjust signs: Π = Q · diag(sign(diag(R)))" },
          ].map((s) => (
            <div key={s.step} className="flex gap-3 items-start">
              <span className="w-6 h-6 rounded-full bg-accent-green/10 text-accent-green text-xs font-bold flex items-center justify-center shrink-0">
                {s.step}
              </span>
              <span className="text-sm text-txt-2">{s.text}</span>
            </div>
          ))}
        </div>
        <div className="mt-6 bg-bg-3 border border-border rounded-xl p-4 text-sm text-txt-2">
          <strong className="text-accent-orange">Trade-off:</strong> O(d²) storage and compute.
          For d=128 (default group size), the rotation matrix is 128×128×4 bytes = 64 KB — manageable.
        </div>
      </Section>

      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm">
          <span className="text-accent-purple">rotation.py</span>{" "}
          <span className="text-txt-2">→</span>{" "}
          <span className="text-accent">generate_rotation_matrix()</span>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
