type NavProps = {
  dark: boolean;
  setDark: (v: boolean) => void;
};

export default function Navbar({ dark, setDark }: NavProps) {
  const wrap = dark
    ? "bg-slate-900/70 border-white/10 backdrop-blur-xl"
    : "bg-white/70 border-white/60 backdrop-blur-xl shadow-sm";

  const title = dark ? "text-white" : "text-slate-900";
  const sub = dark ? "text-slate-400" : "text-slate-500";

  const button = dark
    ? "border-white/20 bg-white/5 text-white hover:bg-white/10"
    : "border-slate-200 bg-white/80 text-slate-800 hover:bg-blue-50";

  return (
    <header className={`sticky top-0 z-50 border-b ${wrap}`}>
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div>
          <h1 className={`text-2xl font-bold tracking-tight ${title}`}>
            AudioInsight
          </h1>

          <p className={`text-xs ${sub}`}>
            Human vs Synthetic Speech Detection
          </p>
        </div>

        <button
          onClick={() => setDark(!dark)}
          className={`rounded-xl border px-4 py-2 text-sm font-medium transition ${button}`}
        >
          {dark ? "Light Mode" : "Dark Mode"}
        </button>
      </div>
    </header>
  );
}