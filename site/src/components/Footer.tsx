export function Footer() {
  return (
    <footer className="border-t border-border py-10 text-center text-sm text-txt-2">
      <div className="max-w-5xl mx-auto px-6">
        <p>
          TurboQuant — Based on{" "}
          <a
            href="https://arxiv.org/abs/2504.19874"
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent hover:underline"
          >
            Zandieh et al., 2025
          </a>{" "}
          · MIT License
        </p>
      </div>
    </footer>
  );
}
