type HeroProps = {
  dark: boolean;
};

export default function Hero({ dark }: HeroProps) {
  const title = dark ? "text-white" : "text-slate-900";
  const sub = dark ? "text-slate-400" : "text-slate-600";

  const chip1 = dark
    ? "bg-blue-500/10 text-blue-300 border-blue-400/20"
    : "bg-blue-50 text-blue-700 border-blue-100";

  const chip2 = dark
    ? "bg-violet-500/10 text-violet-300 border-violet-400/20"
    : "bg-violet-50 text-violet-700 border-violet-100";



  return (
    <section className="relative pt-16 pb-12">
      {!dark && (
        <>
          <div className="absolute -top-8 left-10 h-72 w-72 rounded-full bg-blue-200/40 blur-3xl" />
          <div className="absolute top-8 right-10 h-72 w-72 rounded-full bg-violet-200/40 blur-3xl" />
        </>
      )}

      <div className="relative max-w-4xl">
        <h2
          className={`text-5xl md:text-6xl font-bold leading-tight tracking-tight ${title}`}
        >
          Explainable AI for{" "}
          <span className="bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
            Human vs Synthetic
          </span>{" "}
          Speech Detection
        </h2>

        <p className={`mt-6 max-w-2xl text-lg leading-8 ${sub}`}>
          Compare two deep learning models and understand how they classify uploaded speech samples.
        </p>

        <div className="mt-8 flex flex-wrap gap-3">
          <div className={`rounded-2xl border px-4 py-3 font-medium ${chip1}`}>
            ResNet18
          </div>

          <div className={`rounded-2xl border px-4 py-3 font-medium ${chip2}`}>
            Wav2Vec2
          </div>

          
        </div>
      </div>
    </section>
  );
}
