import { useState } from "react";
import { Toaster, toast } from "sonner";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import UploadCard from "./components/UploadCard";
import ModelPanel from "./components/ModelPanel";
import Interpretation from "./components/Interpretation";
import Footer from "./components/Footer";

const API = import.meta.env.VITE_API_URL || "";

type AnalyzeResult = {
  resnet_prediction: string;
  resnet_score: number;
  w2v_prediction: string;
  w2v_score: number;
  gradcam?: { values: number[]; rows: number; cols: number };
  attention?: { values: number[]; num_frames: number };
  interpretation?: string | null;
};

export default function App() {
  const [dark, setDark] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);

  const pageTheme = dark
    ? "bg-slate-950 text-slate-100"
    : "bg-gradient-to-br from-blue-50 via-white to-violet-50 text-slate-900";

  async function handleAnalyze(file: File) {
    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Analyze failed");
      }

      setResult(data);
    } catch (error: any) {
      console.error(error);
      toast.error(error.message || "Analyze failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`${pageTheme} min-h-screen transition-all duration-500`}>
      <Toaster richColors position="top-right" />

      <Navbar dark={dark} setDark={setDark} />

      <main className="relative mx-auto max-w-7xl px-6 overflow-hidden">
        <Hero dark={dark} />

        <UploadCard
          dark={dark}
          loading={loading}
          onAnalyze={handleAnalyze}
        />

        {loading && (
          <section className="mt-8 rounded-3xl border border-white/20 bg-white/70 p-8 shadow-xl backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
            <div className="flex items-center gap-6">
              <div className="flex h-12 items-end gap-1">
                {[16, 28, 18, 34, 22, 30, 14].map((h, i) => (
                  <div
                    key={i}
                    className="w-2 rounded bg-blue-500 animate-pulse"
                    style={{
                      height: `${h}px`,
                      animationDelay: `${i * 0.1}s`,
                    }}
                  />
                ))}
              </div>

              <div>
                <p className="text-lg font-semibold">Analyzing audio...</p>

                <p className="text-slate-500 dark:text-slate-400">
                  Running ResNet18 + Wav2Vec2 inference
                </p>
              </div>
            </div>
          </section>
        )}

        {result && !loading && (
          <>
            <section className="mt-10 grid gap-6 lg:grid-cols-2">
              <ModelPanel
                dark={dark}
                title="ResNet18"
                subtitle="Mel Spectrogram CNN"
                prediction={result.resnet_prediction}
                score={result.resnet_score}
                note="Spectral artifacts and suspicious frequency regions."
                type="gradcam"
                color="blue"
                visualData={result.gradcam?.values || []}
                rows={result.gradcam?.rows || 8}
                cols={result.gradcam?.cols || 8}
              />

              <ModelPanel
                dark={dark}
                title="Wav2Vec2"
                subtitle="Transformer Speech Encoder"
                prediction={result.w2v_prediction}
                score={result.w2v_score}
                note="Temporal consistency and contextual speech patterns."
                type="attention"
                color="emerald"
                visualData={result.attention?.values || []}
              />
            </section>

            <Interpretation dark={dark} result={result} />
          </>
        )}

        <Footer dark={dark} />
      </main>
    </div>
  );
}