"use client";
import { TechniqueLayout, Section } from "@/components/TechniqueLayout";
import { Math } from "@/components/Math";
import { BitPackingViz } from "@/components/animations/BitPackingViz";

export default function PackingPage() {
  return (
    <TechniqueLayout
      title="4-bit Index Packing"
      subtitle="Two 4-bit indices packed per uint8 byte — halving storage with near-free bitwise operations."
      color="#58a6ff"
      icon="📦"
      prev={{ href: "/techniques/residual/", label: "Residual Quantization" }}
      next={{ href: "/techniques/fused-kernels/", label: "Fused GPU Kernels" }}
    >
      <Section title="Byte Layout">
        <p className="text-txt-2 leading-relaxed mb-6">
          Two 4-bit quantization indices are packed into each uint8 byte. The low nibble
          (bits 3-0) holds index[k] and the high nibble (bits 7-4) holds index[k+1].
        </p>
        <div className="bg-bg-2 border border-border rounded-xl p-6 text-center space-y-4">
          <Math expr="\text{packed}[m, k/2] = \text{idx}[m, k] \;|\; (\text{idx}[m, k+1] \ll 4)" display />
        </div>
      </Section>

      <Section title="Interactive: Pack & Unpack">
        <p className="text-txt-2 leading-relaxed mb-6">
          Watch two 4-bit indices merge into a single byte, then split back apart:
        </p>
        <div className="bg-bg-2 border border-border rounded-2xl p-8">
          <BitPackingViz />
        </div>
      </Section>

      <Section title="Unpack Operations">
        <div className="grid sm:grid-cols-2 gap-4">
          <div className="bg-bg-3 border border-border rounded-xl p-5">
            <h4 className="text-accent font-semibold text-sm mb-2">Low nibble</h4>
            <div className="font-mono text-sm text-txt-2">
              <Math expr="\text{lo} = \text{packed} \;\&\; \texttt{0x0F}" display />
            </div>
          </div>
          <div className="bg-bg-3 border border-border rounded-xl p-5">
            <h4 className="text-accent-purple font-semibold text-sm mb-2">High nibble</h4>
            <div className="font-mono text-sm text-txt-2">
              <Math expr="\text{hi} = (\text{packed} \gg 4) \;\&\; \texttt{0x0F}" display />
            </div>
          </div>
        </div>
      </Section>

      <Section title="Storage Savings">
        <p className="text-txt-2 leading-relaxed">
          An <Math expr="(M, N)" /> index matrix becomes <Math expr="(M, N/2)" /> uint8.
          When <Math expr="N" /> is odd, the last column is zero-padded before packing and the
          original <Math expr="N" /> is stored in metadata for correct unpacking.
        </p>
      </Section>

      <Section title="Implementation">
        <div className="bg-bg-3 border border-border rounded-xl p-5 font-mono text-sm space-y-2">
          <div>
            <span className="text-accent-purple">quantize.py</span>{" "}
            <span className="text-txt-2">→</span>{" "}
            <span className="text-accent">pack_4bit()</span>,{" "}
            <span className="text-accent">unpack_4bit()</span>
          </div>
        </div>
      </Section>
    </TechniqueLayout>
  );
}
