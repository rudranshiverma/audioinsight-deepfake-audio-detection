import { useEffect, useRef } from "react";

type Props = {
  dark: boolean;
  title: string;
  subtitle: string;
  prediction: string;
  score: number;
  note: string;
  type: "gradcam" | "attention";
  color: "blue" | "emerald";
  visualData?: number[];
  rows?: number;
  cols?: number;
};

function jetColor(t: number): [number, number, number] {
  const r = Math.max(0, Math.min(255, Math.round(255 * (1.5 - Math.abs(4 * t - 3)))));
  const g = Math.max(0, Math.min(255, Math.round(255 * (1.5 - Math.abs(4 * t - 2)))));
  const b = Math.max(0, Math.min(255, Math.round(255 * (1.5 - Math.abs(4 * t - 1)))));
  return [r, g, b];
}

//gradcam
function GradCAMCanvas({
  values,
  rows,
  cols,
}: {
  values: number[];
  rows: number;
  cols: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || values.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.clientWidth;
    const H = canvas.clientHeight;

    canvas.width = W;
    canvas.height = H;

    const R = rows || 8;
    const C = cols || 8;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    const cellW = W / C;
    const cellH = H / R;

    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;

    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        const idx = r * C + c;
        const t = ((values[idx] ?? 0) - min) / range;
        const [rv, gv, bv] = jetColor(t);

        ctx.fillStyle = `rgb(${rv},${gv},${bv})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }
  }, [values, rows, cols]);

  if (!values.length) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-400">
        No GradCAM data
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl">
      <canvas ref={canvasRef} className="h-full w-full rounded-xl" />

      <div className="absolute right-2 top-2 bottom-2 w-3 rounded-full overflow-hidden">
        <div
          className="h-full w-full"
          style={{
            background:
              "linear-gradient(to bottom,#ff0000,#ffff00,#00ff00,#00ffff,#0000ff)",
          }}
        />
      </div>
    </div>
  );
}

//attention
function AttentionCanvas({
  values,
  dark,
}: {
  values: number[];
  dark: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas || values.length === 0) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const W = canvas.clientWidth;
      const H = canvas.clientHeight;

      canvas.width = W * window.devicePixelRatio;
      canvas.height = H * window.devicePixelRatio;

      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      ctx.clearRect(0, 0, W, H);

      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min || 1;

      const norm = values.map((v) => (v - min) / range);

      const leftPad = 28;
      const rightPad = 8;
      const topPad = 12;
      const bottomPad = 24;

      const innerW = W - leftPad - rightPad;
      const innerH = H - topPad - bottomPad;

      const n = norm.length;
      const barW = innerW / n;

      for (let i = 0; i < n; i++) {
        const h = Math.max(3, norm[i] * innerH);
        const x = leftPad + i * barW;
        const y = topPad + innerH - h;

        const grad = ctx.createLinearGradient(0, y, 0, topPad + innerH);
        grad.addColorStop(0, "rgba(16,185,129,0.95)");
        grad.addColorStop(1, "rgba(16,185,129,0.20)");

        ctx.fillStyle = grad;
        ctx.fillRect(x + 1, y, barW - 2, h);
      }

      ctx.strokeStyle = dark
        ? "rgba(255,255,255,0.08)"
        : "rgba(0,0,0,0.08)";

      for (let i = 0; i < 4; i++) {
        const y = topPad + (innerH / 3) * i;
        ctx.beginPath();
        ctx.moveTo(leftPad, y);
        ctx.lineTo(W - rightPad, y);
        ctx.stroke();
      }

      ctx.fillStyle = dark
        ? "rgba(148,163,184,0.9)"
        : "rgba(100,116,139,0.9)";

      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";

      ["0s", "1s", "2s", "3s", "4s"].forEach((txt, i) => {
        const x = leftPad + (innerW / 4) * i;
        ctx.fillText(txt, x, H - 6);
      });
    };

    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [values, dark]);

  if (!values.length) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-400">
        No attention data
      </div>
    );
  }

  return <canvas ref={canvasRef} className="h-full w-full" />;
}

//main panel
export default function ModelPanel({
  dark,
  title,
  subtitle,
  prediction,
  score,
  note,
  type,
  color,
  visualData = [],
  rows = 8,
  cols = 8,
}: Props) {
  const isAI = prediction.toLowerCase().includes("ai");

  const confidence = isAI ? score : 1 - score;
  const percent = Math.round(confidence * 100);

  const card = dark
    ? "border border-white/10 bg-white/5 backdrop-blur-xl"
    : "border border-white/70 bg-white/90 backdrop-blur-xl";

  const textMain = dark ? "text-white" : "text-slate-900";
  const textSub = dark ? "text-slate-300" : "text-slate-600";

  const predColor = isAI ? "text-rose-500" : "text-emerald-500";

  const gradient =
    color === "blue"
      ? "from-blue-600 to-cyan-400"
      : "from-emerald-600 to-teal-400";

  return (
    <section className={`rounded-3xl p-6 shadow-xl ${card}`}>
      <div className="flex items-start justify-between">
        <div>
          <h3 className={`text-2xl font-bold ${textMain}`}>{title}</h3>
          <p className={`mt-1 text-sm ${textSub}`}>{subtitle}</p>
        </div>
      </div>

      <div className="mt-6">
        <p className={`text-sm ${textSub}`}>Prediction</p>
        <p className={`mt-1 text-3xl font-bold ${predColor}`}>
          {isAI ? "AI Generated" : "Human"}
        </p>
      </div>

      <div className="mt-6">
        <div className="flex justify-between">
          <p className={`text-sm ${textSub}`}>Confidence</p>
          <p className={`text-sm font-semibold ${textMain}`}>{percent}%</p>
        </div>

        <div className="mt-2 h-3 rounded-full bg-slate-200 overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${gradient}`}
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>

      {/* SAME HEIGHT FOR BOTH */}
      <div
        className={`mt-6 rounded-2xl border p-5 ${
          dark
            ? "border-white/10 bg-black/30"
            : "border-slate-200 bg-slate-50"
        }`}
      >
        <p className={`text-sm font-semibold ${textMain}`}>
          {type === "gradcam"
            ? "Grad-CAM Heatmap"
            : "Attention Weights Over Time"}
        </p>

        <div className="mt-4 h-[260px] w-full">
          {type === "gradcam" ? (
            <GradCAMCanvas
              values={visualData}
              rows={rows}
              cols={cols}
            />
          ) : (
            <AttentionCanvas
              values={visualData}
              dark={dark}
            />
          )}
        </div>
      </div>

      <div
        className={`mt-6 rounded-2xl p-4 ${
          dark ? "bg-black/25" : "bg-slate-50"
        }`}
      >
        <p className={`text-sm leading-6 ${textSub}`}>{note}</p>
      </div>
    </section>
  );
}