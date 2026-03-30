"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { PipelineFlow } from "@/components/animations/PipelineFlow";
import { MatrixCompressionViz } from "@/components/animations/MatrixCompressionViz";
import { Reveal } from "@/components/Reveal";

const quantizeSteps = [
  { icon: "📐", label: "Normalize", detail: "W / ‖W‖₂ → unit norm", color: "#58a6ff" },
  { icon: "🔄", label: "Rotate", detail: "W·Πᵀ → 𝒩(0, 1/d)", color: "#7ee787" },
  { icon: "📏", label: "Scale", detail: "× √d → 𝒩(0, 1)", color: "#d2a8ff" },
  { icon: "🎯", label: "Quantize", detail: "Lloyd-Max → 4-bit idx", color: "#ffa657" },
  { icon: "📦", label: "Pack", detail: "2 indices → 1 byte", color: "#58a6ff" },
];

export default function QuantizePipelinePage() {
  return (
    <TechniqueLayout
      title="Quantization Pipeline"
      subtitle="Compressing each nn.Linear weight matrix W ∈ ℝᴹˣᴺ from bf16/fp32 to 4-bit packed indices in five steps."
      color="#ffa657"
      icon="⚙️"
      prev={{ href: "/techniques/fused-kernels/", label: "Fused Kernels" }}
      next={{ href: "/dequantize-pipeline/", label: "Dequantize Pipeline" }}
    >
      <Section title="Pipeline Overview">
        <PipelineFlow steps={quantizeSteps} />
      </Section>

      <Section title="Step-by-Step">
        <div className="space-y-6">
          <Reveal>
            <StepCard
              num="1"
              title="Row Normalization"
              color="#58a6ff"
              equations={[
                "W_{\\text{norm}}^{(g)} = \\frac{W^{(g)}}{\\|W^{(g)}\\|_2}, \\qquad \\alpha^{(g)} = \\|W^{(g)}\\|_2",
              ]}
            >
              Each row of the group slice is divided by its ℓ₂-norm. The norm α is stored
              separately and applied during inference.
            </StepCard>
          </Reveal>
          <Reveal delay={0.1}>
            <StepCard
              num="2"
              title="Random Rotation"
              color="#7ee787"
              equations={["Y^{(g)} = W_{\\text{norm}}^{(g)} \\cdot \\Pi_g^T"]}
            >
              A random orthogonal matrix Πg decorrelates the weight coordinates. After rotation,
              each coordinate is approximately i.i.d. 𝒩(0, 1/d).
            </StepCard>
          </Reveal>
          <Reveal delay={0.2}>
            <StepCard
              num="3"
              title="Scaling"
              color="#d2a8ff"
              equations={["Y_{\\text{scaled}}^{(g)} = Y^{(g)} \\cdot \\sqrt{d}"]}
            >
              Multiplying by √d brings coordinates to unit variance: 𝒩(0, 1) — exactly matching
              the Lloyd-Max codebook.
            </StepCard>
          </Reveal>
          <Reveal delay={0.3}>
            <StepCard
              num="4"
              title="Scalar Quantization"
              color="#ffa657"
              equations={[
                "\\text{idx}_{m,k} = \\text{searchsorted}(\\text{boundaries}, Y_{\\text{scaled},m,k})",
              ]}
            >
              Each scalar coordinate is independently quantized using the Lloyd-Max optimal
              boundaries for 𝒩(0,1). At 4 bits: 16 centroids, 15 decision boundaries.
            </StepCard>
          </Reveal>
          <Reveal delay={0.4}>
            <StepCard
              num="5"
              title="4-bit Packing"
              color="#58a6ff"
              equations={[
                "\\text{packed}_{m, k/2} = \\text{lo}_k \\;|\\; (\\text{hi}_{k+1} \\ll 4)",
              ]}
            >
              Consecutive pairs of 4-bit indices are packed into a single uint8 byte, halving
              the storage for the index tensor.
            </StepCard>
          </Reveal>
        </div>
      </Section>

      <Section title="Compression Visualization">
        <p className="text-txt-2 mb-6 leading-relaxed">
          Watch the weight matrix transform through each stage of the pipeline:
        </p>
        <div className="bg-bg-2 border border-border rounded-2xl p-6">
          <MatrixCompressionViz />
        </div>
      </Section>

      <Section title="Output Format">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-accent text-xs uppercase">Component</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Shape</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Dtype</th>
                <th className="text-left py-3 px-4 text-xs uppercase text-txt-2">Purpose</th>
              </tr>
            </thead>
            <tbody className="text-txt-2">
              <tr className="border-b border-border"><td className="py-3 px-4 font-mono text-accent">indices_packed</td><td className="py-3 px-4">(M, N/2)</td><td className="py-3 px-4">uint8</td><td className="py-3 px-4">Two 4-bit codebook indices per byte</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4 font-mono text-accent">weight_norms</td><td className="py-3 px-4">(M,) or (M, G)</td><td className="py-3 px-4">float32</td><td className="py-3 px-4">Row or group norms for rescaling</td></tr>
              <tr className="border-b border-border"><td className="py-3 px-4 font-mono text-accent">codebook</td><td className="py-3 px-4">(2ᵇ,)</td><td className="py-3 px-4">float32</td><td className="py-3 px-4">Lloyd-Max centroids (shared globally)</td></tr>
              <tr><td className="py-3 px-4 font-mono text-accent">seed</td><td className="py-3 px-4">scalar</td><td className="py-3 px-4">int</td><td className="py-3 px-4">Rotation seed for reproducibility</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Implementation Entry Points">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">quantize.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">turboquant_quantize_packed()</span>{" "}
            <span className="text-txt-2">(standalone)</span>
          </div>
          <div>
            <span className="text-accent-purple">model.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">quantize_model()</span>{" "}
            <span className="text-txt-2">(full model)</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}

function StepCard({
  num,
  title,
  color,
  equations,
  children,
}: {
  num: string;
  title: string;
  color: string;
  equations: string[];
  children: React.ReactNode;
}) {
  return (
    <div className="bg-bg-2 border border-border rounded-xl p-6">
      <div className="flex items-start gap-4">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold shrink-0"
          style={{ background: `${color}15`, color }}
        >
          {num}
        </div>
        <div className="flex-1">
          <h3 className="font-semibold mb-2">{title}</h3>
          <p className="text-sm text-txt-2 leading-relaxed mb-3">{children}</p>
          {equations.map((eq, i) => (
            <div key={i} className="bg-bg-3 rounded-lg p-3 text-center mt-2">
              <Math expr={eq} display />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
