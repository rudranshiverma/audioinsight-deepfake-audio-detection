//uploadcard.tsx
import { useState } from "react";

type UploadProps = {
  dark: boolean;
  loading: boolean;
  onAnalyze: (file: File) => void;
};

export default function UploadCard({
  dark,
  loading,
  onAnalyze,
}: UploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState("");

  const card = dark
    ? "bg-white/5 border-white/10 backdrop-blur-xl"
    : "bg-white/75 border-white/60 backdrop-blur-xl shadow-2xl shadow-blue-100/50";

  const title = dark ? "text-white" : "text-slate-900";
  const sub = dark ? "text-slate-400" : "text-slate-600";

  const drop = dark
    ? "border-white/10 bg-black/20 hover:bg-black/30"
    : "border-blue-200 bg-gradient-to-br from-blue-50 to-violet-50 hover:from-blue-100 hover:to-violet-100";

  const button = dark
    ? "bg-white text-slate-900 hover:bg-slate-200"
    : "bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:opacity-95 shadow-lg shadow-blue-200";

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setAudioUrl(URL.createObjectURL(file));
  }

  return (
    <section className={`rounded-3xl border p-8 transition ${card}`}>
      <div className="mx-auto max-w-2xl text-center">
        <h3 className={`text-3xl font-bold ${title}`}>
          Upload Audio File
        </h3>

        <p className={`mt-2 text-sm ${sub}`}>
          Supported formats: WAV, MP3, FLAC
        </p>

        <label
          className={`mt-6 flex min-h-[160px] cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed px-6 text-center transition ${drop}`}
        >
          <input
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a"
            className="hidden"
            onChange={handleFile}
          />

          <p className={`text-xl font-semibold ${title}`}>
            Click to Upload
          </p>

          <p className={`mt-2 text-sm ${sub}`}>
            Clear speech clips recommended
          </p>
        </label>

        {selectedFile && (
          <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-left">
            <p className="text-sm font-medium text-emerald-700">
              {selectedFile.name}
            </p>
          </div>
        )}

        {audioUrl && (
          <div
            className={`mt-4 rounded-2xl p-4 ${
              dark ? "bg-black/20" : "bg-white/80 shadow-md"
            }`}
          >
            <audio controls src={audioUrl} className="w-full" />
          </div>
        )}

        <button
          disabled={!selectedFile || loading}
          onClick={() => selectedFile && onAnalyze(selectedFile)}
          className={`mt-6 rounded-2xl px-7 py-3 font-semibold transition disabled:opacity-50 ${button}`}
        >
          {loading ? "Analyzing..." : "Analyze Audio"}
        </button>
      </div>
    </section>
  );
}