type FootProps = {
  dark: boolean;
};
export default function Footer({ dark }: FootProps) {
  const border = dark ? "border-white/10" : "border-slate-200";
  const text = dark ? "text-slate-300" : "text-slate-500";

  return (
    <footer className={`mt-16 border-t py-10 ${border}`}>
      <div className={`text-center text-sm ${text}`}>
        AudioInsight • React • Tailwind • PyTorch • Explainable AI
      </div>
    </footer>
  );
}