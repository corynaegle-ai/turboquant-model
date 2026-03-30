"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { ButterflyDiagram } from "@/components/animations/ButterflyDiagram";
import { Reveal } from "@/components/Reveal";

export default function WalshHadamardPage() {
  return (
    <TechniqueLayout
      title="Walsh-Hadamard Transform"
      subtitle="O(d log d) butterfly-style rotation with O(d) storage — the fast alternative to QR decomposition for random rotations."
      color="#d2a8ff"
      icon="⚡"
      prev={{ href: "/techniques/rotation/", label: "Random Rotation" }}
      next={{ href: "/techniques/residual/", label: "Residual Quantization" }}
    >
      <Section title="Definition">
        <p className="text-txt-2 leading-relaxed mb-6">
          The Walsh-Hadamard matrix <Math expr="H_n" /> of order{" "}
          <Math expr="n = 2^k" /> is defined recursively:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center">
          <Math
            expr="H_1 = [1], \qquad H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed mt-4">
          The normalized Hadamard matrix{" "}
          <Math expr="\bar{H} = H / \sqrt{d}" /> is orthogonal:{" "}
          <Math expr="\bar{H}^T \bar{H} = I" />.
        </p>
      </Section>

      <Section title="Butterfly Diagram">
        <p className="text-txt-2 leading-relaxed mb-6">
          Analogous to FFT, the FWHT computes <Math expr="y = H \cdot x" /> in{" "}
          <Math expr="O(d \log d)" /> time without materializing the full matrix. The butterfly
          pattern shows how elements combine at each stage:
        </p>
        <div className="bg-bg-2 border border-border rounded-2xl p-8">
          <ButterflyDiagram />
        </div>
      </Section>

      <Section title="Randomized Hadamard Rotation">
        <p className="text-txt-2 leading-relaxed mb-4">
          A plain Hadamard matrix is deterministic. To get a random rotation, TurboQuant uses:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\Pi = \frac{1}{\sqrt{d}} H \cdot D \quad \text{where } D = \text{diag}(s_1, \ldots, s_d), \ s_i \in \{-1, +1\}"
            display
          />
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          <div className="bg-bg-3 border border-border rounded-xl p-4">
            <h4 className="font-semibold text-accent-green text-sm mb-2">Forward rotation</h4>
            <Math expr="Y = \text{FWHT}(X \odot \mathbf{s}) / \sqrt{d}" display />
          </div>
          <div className="bg-bg-3 border border-border rounded-xl p-4">
            <h4 className="font-semibold text-accent-orange text-sm mb-2">Inverse rotation</h4>
            <Math expr="X = \text{FWHT}(Y) / \sqrt{d} \odot \mathbf{s}" display />
          </div>
        </div>
      </Section>

      <Section title="QR vs Hadamard Comparison">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent text-xs uppercase tracking-wide">Property</th>
                <th className="text-left py-3 px-4 text-xs uppercase tracking-wide text-txt-2">QR (Haar)</th>
                <th className="text-left py-3 px-4 text-xs uppercase tracking-wide text-accent-green">Hadamard</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border"><td className="py-3 px-4">Storage</td><td className="py-3 px-4">O(d²) — full matrix</td><td className="py-3 px-4 text-accent-green font-semibold">O(d) — just sign vector</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4">Compute</td><td className="py-3 px-4">O(d²) — matrix multiply</td><td className="py-3 px-4 text-accent-green font-semibold">O(d log d) — FWHT</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4">Randomness</td><td className="py-3 px-4">Exact Haar distribution</td><td className="py-3 px-4">Approximate (excellent)</td></tr>
              <tr><td className="py-3 px-4">Constraint</td><td className="py-3 px-4">None</td><td className="py-3 px-4">d must be power of 2</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Benchmark: Identical Quality">
        <div className="bg-bg-2 border border-accent-green/20 rounded-xl p-6">
          <p className="text-txt-2 leading-relaxed text-sm">
            On Qwen3.5-0.8B, Hadamard rotation matches QR quality exactly:
            PPL 14.30 vs 14.28 (4+4 residual), KLD 0.0020 for both. Use{" "}
            <code className="text-accent-purple">--rotation hadamard</code> to enable.
          </p>
        </div>
      </Section>

      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">rotation.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">hadamard_rotate()</span>,{" "}
            <span className="text-accent">hadamard_rotate_inverse()</span>,{" "}
            <span className="text-accent">_fwht()</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
