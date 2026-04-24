//interpretation.tsx
type AnalyzeResult = {
  resnet_prediction: string;
  resnet_score: number;
  w2v_prediction: string;
  w2v_score: number;
  gradcam?: { values:number[]; rows:number; cols:number };
  attention?: { values:number[]; num_frames:number };
  interpretation?: string | null;
};
type Props = {
  dark: boolean;
  result: AnalyzeResult;
};
export default function Interpretation({ dark, result }: Props) {
  const card = dark
    ? "bg-slate-900 border-slate-800"
    : "bg-white border-slate-200";

  const subCard = dark
    ? "bg-slate-950"
    : "bg-slate-50";

  const textMain = dark ? "text-slate-100" : "text-slate-900";
  const textSub = dark ? "text-slate-400" : "text-slate-600";

const resnetScore = result.resnet_score;
const w2vScore = result.w2v_score;
const r = result.resnet_prediction;
const w = result.w2v_prediction;

//agreement level
const bothAI = r === "AI" && w === "AI";
const bothHuman = r === "Human" && w === "Human";

//confidence levels
const resnetConfidence = r === "AI" ? resnetScore : 1 - resnetScore;
const w2vConfidence = w === "AI" ? w2vScore : 1 - w2vScore;
const avgConfidence = (resnetConfidence + w2vConfidence) / 2;

const highConf = avgConfidence > 0.85;
const lowConf = avgConfidence < 0.60;
const modelsDiverge = Math.abs(resnetConfidence - w2vConfidence) > 0.35;

let body = "";
let verdict = "";

if (bothAI && highConf) {
  verdict = "Strong synthetic signal detected.";
  body = "Both models agree with high confidence. ResNet18 identified suspicious patterns in the frequency spectrum while Wav2Vec2 detected unnatural temporal transitions. High agreement between complementary feature spaces strongly suggests AI-generated audio.";
} else if (bothAI && lowConf) {
  verdict = "Likely synthetic, low certainty.";
  body = "Both models lean toward synthetic origin but with limited confidence. The audio may be a high-quality TTS system that partially preserves natural speech characteristics, or a human recording with unusual acoustic properties.";
  
} else if (bothAI) {
  verdict = "Synthetic audio detected.";
  body = "Both models independently identified synthetic cues — ResNet18 through spectral analysis and Wav2Vec2 through temporal pattern analysis. Dual-model agreement increases detection confidence.";
  
} else if (bothHuman && highConf) {
  verdict = "Strong human speech signal.";
  body = "Both models detected natural acoustic and temporal patterns with high confidence. The frequency structure and speech flow are consistent with authentic human speech.";
  
} else if (bothHuman && lowConf) {
  verdict = "Likely human, borderline case.";
  body = "Both models lean toward human but with limited confidence. This may indicate a high-quality voice clone that preserves natural prosody, or background noise affecting the analysis. Consider testing a longer or cleaner audio clip.";
  
} else if (bothHuman) {
  verdict = "Human speech detected.";
  body = "Both models found natural speech characteristics. No significant synthetic artifacts were identified in either the spectral or temporal domain.";
  
} else if (r === "AI" && w === "Human" && modelsDiverge) {
  verdict = "Conflicting signals — spectral anomaly detected.";
  body = "ResNet18 detected suspicious frequency-domain artifacts that are characteristic of neural vocoders, while Wav2Vec2 found natural temporal flow. This specific pattern is common in TTS systems that preserve natural prosody but leave traces in the mel spectrogram. ResNet18's signal is more diagnostic in this case.";
  
} else if (r === "AI" && w === "Human") {
  verdict = "Mild spectral anomaly, temporal patterns normal.";
  body = "A weak synthetic signal was detected in the frequency domain. Wav2Vec2 found no temporal inconsistencies. This could indicate light post-processing on a human recording, or a well-crafted TTS with natural prosody.";
  
} else if (r === "Human" && w === "AI" && modelsDiverge) {
  verdict = "Conflicting signals — temporal anomaly detected.";
  body = "Wav2Vec2 detected unnatural transitions in the speech flow over time, while ResNet18 found normal spectral characteristics. This pattern can occur with voice conversion systems that modify timing without altering fundamental frequency structure.";
  
} else {
  verdict = "Mild temporal anomaly, spectral patterns normal.";
  body = "A weak synthetic signal was detected in the temporal domain. ResNet18 found no spectral anomalies. The result is ambiguous — the audio has mostly natural characteristics with minor irregularities in speech timing.";
  
}

  return (
    <section className="mt-8 space-y-6">
      {/* Interpretation */}
      <div className={`rounded-3xl border p-8 shadow-sm ${card}`}>
        <h3 className={`text-2xl font-bold ${textMain}`}>
          Interpretation
        </h3>
        <h5 className={`mt-4 max-w-4xl leading-8 ${textSub}`}>
          {result.interpretation || verdict}
        </h5>
        <p className={`mt-4 max-w-4xl leading-8 ${textSub}`}>
          {result.interpretation || body}
        </p>
      </div>

      {/* Model Info */}
      <div className={`rounded-3xl border p-8 shadow-sm ${card}`}>
        <h3 className={`text-2xl font-bold ${textMain}`}>
          Model Info
        </h3>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <div className={`rounded-2xl p-5 ${subCard}`}>
            <p className={`font-semibold ${textMain}`}>
              ResNet18
            </p>

            <p className={`mt-2 text-sm leading-6 ${textSub}`}>
              Specialized in spectrogram pattern recognition and
              frequency-time anomaly detection.
            </p>
          </div>

          <div className={`rounded-2xl p-5 ${subCard}`}>
            <p className={`font-semibold ${textMain}`}>
              Wav2Vec2
            </p>

            <p className={`mt-2 text-sm leading-6 ${textSub}`}>
              Learns speech representations over time and captures
              temporal consistency cues.
            </p>
          </div>

          <div className={`rounded-2xl p-5 ${subCard}`}>
            <p className={`font-semibold ${textMain}`}>
              Combined Value
            </p>

            <p className={`mt-2 text-sm leading-6 ${textSub}`}>
              Together they provide complementary evidence for
              human vs synthetic speech analysis.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}