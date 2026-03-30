"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { GaussianQuantizationViz } from "@/components/animations/GaussianQuantizationViz";
import { CodebookViz } from "@/components/animations/CodebookViz";
import { Reveal } from "@/components/Reveal";

export default function LloydMaxPage() {
  return (
    <TechniqueLayout
      title="Lloyd-Max Scalar Quantization"
      subtitle="The optimal scalar quantizer for Gaussian distributions. Iteratively refines centroids and decision boundaries to minimize mean squared error."
      color="#58a6ff"
      icon="📊"
      prev={{ href: "/", label: "Home" }}
      next={{ href: "/techniques/rotation/", label: "Random Rotation" }}
    >
      <Section title="The Problem">
        <p className="text-txt-2 leading-relaxed mb-6">
          Given a continuous random variable <Math expr="X \sim p(x)" /> and a budget of{" "}
          <Math expr="L = 2^b" /> levels (16 at 4 bits), find reconstruction values (centroids){" "}
          <Math expr="c_1, \ldots, c_L" /> and decision boundaries{" "}
          <Math expr="t_0 < t_1 < \cdots < t_L" /> that minimize mean squared error:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center">
          <Math
            expr="\text{MSE} = \mathbb{E}\left[(X - Q(X))^2\right] = \int_{-\infty}^{\infty} (x - Q(x))^2 \, p(x)\, dx"
            display
          />
        </div>
      </Section>

      <Section title="The Algorithm">
        <p className="text-txt-2 leading-relaxed mb-6">
          The Lloyd-Max algorithm (Lloyd 1982, Max 1960) iteratively refines centroids and
          boundaries until convergence (~200 iterations):
        </p>
        <div className="space-y-4">
          {[
            {
              step: "1",
              title: "Initialize",
              desc: "Place centroids uniformly across the distribution's range",
            },
            {
              step: "2",
              title: "Update boundaries (nearest-neighbor rule)",
              math: "t_i = \\frac{c_i + c_{i+1}}{2}",
            },
            {
              step: "3",
              title: "Update centroids (conditional expectation)",
              math: "c_i = \\frac{\\int_{t_{i-1}}^{t_i} x \\, p(x) \\, dx}{\\int_{t_{i-1}}^{t_i} p(x) \\, dx}",
            },
            { step: "4", title: "Repeat", desc: "Until MSE converges" },
          ].map((s, i) => (
            <Reveal key={s.step} delay={0.1 * i}>
              <div className="flex gap-4 items-start bg-bg-2 border border-border rounded-xl p-5">
                <div className="w-8 h-8 rounded-full bg-accent/10 flex items-center justify-center text-accent font-bold text-sm shrink-0">
                  {s.step}
                </div>
                <div>
                  <div className="font-semibold mb-1">{s.title}</div>
                  {s.desc && <p className="text-sm text-txt-2">{s.desc}</p>}
                  {s.math && (
                    <div className="mt-2">
                      <Math expr={s.math} display />
                    </div>
                  )}
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section title="Interactive Visualization">
        <p className="text-txt-2 leading-relaxed mb-6">
          The Gaussian curve below shows the 16 Lloyd-Max centroids (vertical lines) and
          decision boundaries for <Math expr="\mathcal{N}(0, 1)" />. Each centroid is the
          conditional mean of its partition region — placing more levels where the density is
          highest.
        </p>
        <div className="bg-bg-2 border border-border rounded-2xl p-6">
          <GaussianQuantizationViz />
        </div>
      </Section>

      <Section title="The 16-Entry Codebook">
        <p className="text-txt-2 leading-relaxed mb-6">
          At 4 bits, the codebook is just 16 float32 values (64 bytes) — shared across
          all layers. This is <strong className="text-txt">optimal</strong>: no other scalar
          quantizer with the same number of levels achieves lower MSE for the Gaussian
          distribution.
        </p>
        <CodebookViz />
      </Section>

      <Section title="Why It Works for TurboQuant">
        <div className="bg-bg-2 border border-accent/20 rounded-xl p-6">
          <p className="text-txt-2 leading-relaxed">
            After rotation, each weight coordinate is approximately{" "}
            <Math expr="\mathcal{N}(0, 1)" />. The Lloyd-Max codebook for this distribution is
            computed once and reused everywhere — achieving overall distortion within{" "}
            <strong className="text-accent">2.7× of the information-theoretic lower bound</strong>{" "}
            (Shannon rate-distortion).
          </p>
        </div>
      </Section>

      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm">
          <span className="text-accent-purple">codebook.py</span>{" "}
          <span className="text-txt-2">→</span>{" "}
          <span className="text-accent">_compute_lloyd_max_gaussian()</span>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
