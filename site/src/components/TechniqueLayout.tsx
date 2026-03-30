"use client";
import { Reveal } from "./Reveal";
import Link from "next/link";
import { ReactNode } from "react";

interface TechniqueLayoutProps {
  title: string;
  subtitle: string;
  color: string;
  icon: string;
  children: ReactNode;
  prev?: { href: string; label: string };
  next?: { href: string; label: string };
}

export function TechniqueLayout({
  title,
  subtitle,
  color,
  icon,
  children,
  prev,
  next,
}: TechniqueLayoutProps) {
  return (
    <div className="pt-20 pb-24">
      {/* Hero */}
      <section className="py-20 relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-10 blur-3xl"
            style={{ background: color }}
          />
        </div>
        <div className="max-w-4xl mx-auto px-6 relative">
          <Reveal>
            <div className="text-5xl mb-4">{icon}</div>
            <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
              <span
                className="bg-clip-text text-transparent"
                style={{
                  backgroundImage: `linear-gradient(135deg, ${color}, #e6edf3)`,
                }}
              >
                {title}
              </span>
            </h1>
            <p className="text-lg text-txt-2 max-w-2xl">{subtitle}</p>
          </Reveal>
        </div>
      </section>

      {/* Content */}
      <div className="max-w-4xl mx-auto px-6 space-y-16">{children}</div>

      {/* Navigation */}
      <div className="max-w-4xl mx-auto px-6 mt-24">
        <div className="flex justify-between items-center border-t border-border pt-8">
          {prev ? (
            <Link
              href={prev.href}
              className="text-sm text-txt-2 hover:text-accent transition-colors"
            >
              ← {prev.label}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              href={next.href}
              className="text-sm text-txt-2 hover:text-accent transition-colors"
            >
              {next.label} →
            </Link>
          ) : (
            <span />
          )}
        </div>
      </div>
    </div>
  );
}

/* Section block for within technique pages */
export function Section({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Reveal className={className}>
      {title && (
        <h2 className="text-2xl font-bold mb-6 text-txt">{title}</h2>
      )}
      {children}
    </Reveal>
  );
}
