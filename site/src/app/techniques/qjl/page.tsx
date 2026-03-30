"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { Reveal } from "@/components/Reveal";

export default function QjlPage() {
  return (
    <TechniqueLayout
      title="Why Not QJL?"
      subtitle="The TurboQuant paper defines TurboQuant_prod using QJL (Quantized Johnson-Lindenstrauss) — but this project deliberately does not use it. Here's why."
      color="#f85149"
      icon="🚫"
      prev={{ href: "/techniques/fused-kernels/", label: "Fused Kernels" }}
      next={{ href: "/quantize-pipeline/", label: "Quantize Pipeline" }}
    >
      <Section title="What Is QJL?">
        <p className="text-txt-2 leading-relaxed mb-4">
          QJL (Quantized Johnson-Lindenstrauss) is based on the Johnson-Lindenstrauss lemma,
          which states that high-dimensional points can be projected into much lower dimensions
          while approximately preserving pairwise distances.
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center mb-6">
          <Math
            expr="\hat{\langle q, k \rangle} = \frac{1}{m} \sum_{i=1}^{m} \text{sign}(\langle r_i, q \rangle) \cdot \text{sign}(\langle r_i, k \rangle) \cdot \|q\| \cdot \|k\|"
            display
          />
        </div>
        <p className="text-txt-2 leading-relaxed">
          QJL projects vectors onto random directions and keeps only the <strong className="text-txt">sign</strong> (1 bit per
          projection). This produces an <strong className="text-txt">unbiased estimator</strong>{" "}
          of the inner product — the expected value equals the true dot product.
        </p>
      </Section>

      <Section title="QJL in the TurboQuant Paper">
        <p className="text-txt-2 leading-relaxed mb-4">
          The paper defines <strong className="text-accent-purple">TurboQuant<sub>prod</sub></strong>:
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 space-y-3">
          <div className="flex gap-3 items-start">
            <span className="w-6 h-6 rounded-full bg-accent-purple/10 text-accent-purple text-xs font-bold flex items-center justify-center shrink-0">1</span>
            <span className="text-sm text-txt-2">Quantize the vector using TurboQuant (rotation + Lloyd-Max)</span>
          </div>
          <div className="flex gap-3 items-start">
            <span className="w-6 h-6 rounded-full bg-accent-purple/10 text-accent-purple text-xs font-bold flex items-center justify-center shrink-0">2</span>
            <span className="text-sm text-txt-2">Compute the residual error</span>
          </div>
          <div className="flex gap-3 items-start">
            <span className="w-6 h-6 rounded-full bg-accent-purple/10 text-accent-purple text-xs font-bold flex items-center justify-center shrink-0">3</span>
            <span className="text-sm text-txt-2">Apply <strong className="text-accent-purple">1-bit QJL</strong> to the residual for an unbiased correction term</span>
          </div>
        </div>
        <p className="text-txt-2 leading-relaxed mt-4">
          This makes the overall inner product estimator <strong className="text-txt">unbiased</strong> — useful for
          KV-cache attention where you quantize keys once and query with many different vectors.
        </p>
      </Section>

      <Section title="Four Reasons We Don't Use QJL">
        <div className="space-y-4">
          <Reveal>
            <ReasonCard
              num="1"
              title="QJL Solves a Different Problem"
              color="#f85149"
            >
              QJL is designed for <strong className="text-txt">online inner product estimation</strong> — quantize
              once, query many times with different vectors. Weight quantization is{" "}
              <strong className="text-txt">offline</strong>: we compress <Math expr="W" /> once and compute{" "}
              <Math expr="y = xW^T" /> repeatedly. We want minimum reconstruction error{" "}
              <Math expr="\|W - \tilde{W}\|" />, not an unbiased dot-product estimator.
            </ReasonCard>
          </Reveal>

          <Reveal delay={0.1}>
            <ReasonCard
              num="2"
              title="Unbiasedness Is Unnecessary for Weights"
              color="#ffa657"
            >
              A small deterministic bias from MSE-optimal quantization is absorbed by layer norms,
              residual connections, and softmax normalization. An unbiased but{" "}
              <strong className="text-txt">high-variance</strong> estimator (QJL at 1 bit) introduces
              stochastic noise that changes every forward pass — worse for stable inference.
            </ReasonCard>
          </Reveal>

          <Reveal delay={0.2}>
            <ReasonCard
              num="3"
              title="Residual Quantization Strictly Dominates"
              color="#7ee787"
            >
              <p>
                QJL uses <strong className="text-txt">1 bit</strong> (random sign projection) for the
                residual correction. Our residual pass uses <Math expr="b_2" /> bits with a full
                Lloyd-Max codebook + independent rotation — capturing far more residual information.
              </p>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <div className="bg-bg-3 rounded-lg p-3 text-center">
                  <div className="text-sm font-bold" style={{ color: "#f85149" }}>QJL correction</div>
                  <div className="text-xs text-txt-2 mt-1">1 bit per weight</div>
                  <div className="text-xs text-txt-2">Random sign only</div>
                </div>
                <div className="bg-bg-3 rounded-lg p-3 text-center">
                  <div className="text-sm font-bold text-accent-green">Residual TQ</div>
                  <div className="text-xs text-txt-2 mt-1">4 bits per weight</div>
                  <div className="text-xs text-txt-2">Full Lloyd-Max codebook</div>
                </div>
              </div>
              <p className="mt-3">
                At 4+4 total bits, residual TurboQuant achieves KL divergence of only{" "}
                <strong className="text-accent-green">0.002 nats</strong> (practically lossless). A 1-bit QJL
                correction cannot compete.
              </p>
            </ReasonCard>
          </Reveal>

          <Reveal delay={0.3}>
            <ReasonCard
              num="4"
              title="QJL Requires the Query at Runtime"
              color="#d2a8ff"
            >
              The QJL correction term depends on the input activation <Math expr="x" />, making it
              incompatible with offline weight compression. You&apos;d need to recompute corrections per
              forward pass — defeating the purpose of weight-only quantization.
            </ReasonCard>
          </Reveal>
        </div>
      </Section>

      <Section title="Visual Comparison">
        <div className="grid md:grid-cols-2 gap-6">
          <Reveal>
            <div className="bg-bg-2 border border-border rounded-xl p-6 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-red-500/50 to-transparent" />
              <h4 className="font-semibold text-sm mb-4" style={{ color: "#f85149" }}>
                TurboQuant<sub>prod</sub> (Paper)
              </h4>
              <div className="space-y-2 text-xs text-txt-2">
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span className="text-accent-purple">Pass 1:</span> Lloyd-Max quantize (b₁ bits)
                </div>
                <div className="text-center text-accent text-sm">+</div>
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span style={{ color: "#f85149" }}>Pass 2:</span> QJL 1-bit sign projection on residual
                </div>
                <div className="text-center text-accent text-sm">↓</div>
                <div className="bg-bg-3 rounded-lg p-2.5">
                  <strong className="text-txt">Unbiased</strong> inner product estimator.
                  Needs query x at runtime.
                </div>
              </div>
            </div>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="bg-bg-2 border border-accent-green/30 rounded-xl p-6 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-accent-green/70 to-transparent" />
              <h4 className="text-accent-green font-semibold text-sm mb-4">
                This Project (Residual TQ)
              </h4>
              <div className="space-y-2 text-xs text-txt-2">
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span className="text-accent-green">Pass 1:</span> Full TQ: rotate + Lloyd-Max (4 bits)
                </div>
                <div className="text-center text-accent text-sm">+</div>
                <div className="bg-bg-3 rounded-lg p-2.5 flex items-center gap-2">
                  <span className="text-accent-green">Pass 2:</span> Full TQ on residual (4 bits, new codebook)
                </div>
                <div className="text-center text-accent text-sm">↓</div>
                <div className="bg-bg-3 rounded-lg p-2.5">
                  <strong className="text-accent-green">Near-lossless</strong> weight compression.
                  Offline, no runtime dependency.
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </Section>

      <Section title="Summary">
        <div className="bg-bg-2 border border-accent-green/20 rounded-xl p-6">
          <p className="text-txt-2 leading-relaxed">
            QJL is elegant for streaming / KV-cache inner product preservation. For{" "}
            <strong className="text-txt">weight compression</strong>, multi-pass residual
            quantization with optimal scalar codebooks is the natural and superior choice — achieving
            practically lossless results at 4+4 bits with no runtime overhead.
          </p>
        </div>
      </Section>

      <Section title="References">
        <div className="space-y-2 text-sm text-txt-2">
          <p>
            <strong className="text-txt">QJL:</strong> Zandieh et al., &quot;QJL: 1-Bit Quantized JL
            Transform for KV Cache Quantization with Zero Overhead,&quot; 2024.
          </p>
          <p>
            <strong className="text-txt">Johnson-Lindenstrauss:</strong> W. Johnson &amp; J. Lindenstrauss,
            &quot;Extensions of Lipschitz mappings into a Hilbert space,&quot; Contemporary Mathematics, 1984.
          </p>
        </div>
      </Section>
    </TechniqueLayout>
  );
}

function ReasonCard({
  num,
  title,
  color,
  children,
}: {
  num: string;
  title: string;
  color: string;
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
        <div>
          <h3 className="font-semibold mb-2">{title}</h3>
          <div className="text-sm text-txt-2 leading-relaxed">{children}</div>
        </div>
      </div>
    </div>
  );
}
