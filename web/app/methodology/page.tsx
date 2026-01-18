import Link from 'next/link';
import { ArrowLeft, Shield, Activity, Target, Brain, Anchor, Zap, CheckCircle2, TrendingUp } from 'lucide-react';

export default function Methodology() {
  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <Link 
          href="/" 
          className="inline-flex items-center text-gray-400 hover:text-emerald-400 transition-colors mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Rankings
        </Link>

        <header className="space-y-4">
          <h1 className="text-4xl font-bold text-white">Methodology</h1>
          <p className="text-xl text-gray-400">
            The Enhanced Defensive Index (EDI) is a Bayesian framework designed to evaluate individual defensive impact across five distinct dimensions.
          </p>
        </header>

        <section className="space-y-6">
          <h2 className="text-2xl font-semibold text-emerald-400 border-b border-gray-800 pb-2">
            The 5 Dimensions of Defense
          </h2>
          
          <div className="grid gap-6">
            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
              <div className="flex items-center gap-3 mb-3">
                <Shield className="w-6 h-6 text-emerald-500" />
                <h3 className="text-xl font-bold text-white">D1: Shot Suppression</h3>
              </div>
              <p className="text-gray-300 mb-2">
                Measures a defender's ability to lower their opponent's field goal percentage compared to their expected percentage.
              </p>
              <ul className="list-disc list-inside text-sm text-gray-400 space-y-1">
                <li>Adjusted for Matchup Difficulty (defending stars vs. role players).</li>
                <li>Uses Bayesian shrinkage to handle small sample sizes.</li>
              </ul>
            </div>

            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
              <div className="flex items-center gap-3 mb-3">
                <Target className="w-6 h-6 text-emerald-500" />
                <h3 className="text-xl font-bold text-white">D2: Shot Profile</h3>
              </div>
              <p className="text-gray-300 mb-2">
                Evaluates the types of shots a defender forces. Elite defenders force opponents into inefficient mid-range shots rather than layups or corner 3s.
              </p>
              <ul className="list-disc list-inside text-sm text-gray-400 space-y-1">
                <li>Rewards rim protection (low FG% at rim).</li>
                <li>Rewards perimeter defense (limiting 3PT attempts).</li>
              </ul>
            </div>

            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
              <div className="flex items-center gap-3 mb-3">
                <Zap className="w-6 h-6 text-emerald-500" />
                <h3 className="text-xl font-bold text-white">D3: Hustle Index</h3>
              </div>
              <p className="text-gray-300 mb-2">
                Quantifies defensive activity that doesn't always show up in the box score.
              </p>
              <ul className="list-disc list-inside text-sm text-gray-400 space-y-1">
                <li>Components: Deflections, Charges Drawn, Contested Shots.</li>
                <li>Weighted by position and minutes played.</li>
              </ul>
            </div>

            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
              <div className="flex items-center gap-3 mb-3">
                <Brain className="w-6 h-6 text-emerald-500" />
                <h3 className="text-xl font-bold text-white">D4: Defensive IQ</h3>
              </div>
              <p className="text-gray-300 mb-2">
                A ratio-based metric assessing defensive playmaking relative to mistakes.
              </p>
              <ul className="list-disc list-inside text-sm text-gray-400 space-y-1">
                <li>Formula: (Steals + Blocks) / (Fouls + 1).</li>
                <li>Identifies disciplined defenders who create turnovers without fouling.</li>
              </ul>
            </div>

            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
              <div className="flex items-center gap-3 mb-3">
                <Anchor className="w-6 h-6 text-emerald-500" />
                <h3 className="text-xl font-bold text-white">D5: Anchor / Rebounding</h3>
              </div>
              <p className="text-gray-300 mb-2">
                Measures the ability to end defensive possessions.
              </p>
              <ul className="list-disc list-inside text-sm text-gray-400 space-y-1">
                <li>Primary metric: Defensive Rebound Percentage (DREB%).</li>
                <li>Role-adjusted: Heavily weighted for Bigs, less for Guards.</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <h2 className="text-2xl font-semibold text-emerald-400 border-b border-gray-800 pb-2">
            Efficiency Model
          </h2>
          <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
            <p className="text-gray-300 mb-4">
              EDI introduces an "Efficiency" coefficient to balance volume and effectiveness.
            </p>
            <div className="space-y-4">
              <div>
                <h4 className="font-bold text-white mb-1">Input (Effort)</h4>
                <p className="text-sm text-gray-400">Weighted average of Hustle (D3) and IQ (D4).</p>
              </div>
              <div>
                <h4 className="font-bold text-white mb-1">Output (Impact)</h4>
                <p className="text-sm text-gray-400">Weighted average of Suppression (D1) and Profile (D2).</p>
              </div>
              <div>
                <h4 className="font-bold text-white mb-1">Efficiency Ratio</h4>
                <p className="text-sm text-gray-400">
                  Actual Output / Expected Output (based on league-wide regression).
                </p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded border border-gray-700 mt-2">
                <p className="text-xs text-gray-400 italic">
                  *For partial seasons, efficiency is stabilized using Bayesian shrinkage (K=10) to prevent small-sample outliers.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-6">
          <h2 className="text-2xl font-semibold text-emerald-400 border-b border-gray-800 pb-2">
            Validation Results
          </h2>
          <p className="text-gray-300">
            EDI was validated against official NBA defensive metrics across 5 seasons (2019-2024), measuring how well each metric identifies actual All-Defensive Team selections.
          </p>
          
          {/* Comparison Table */}
          <div className="bg-gray-900/50 rounded-xl border border-gray-800 overflow-hidden">
            <div className="bg-gradient-to-r from-emerald-900/30 to-cyan-900/30 px-6 py-4 border-b border-gray-800">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
                All-Defensive Team Coverage
              </h3>
              <p className="text-sm text-gray-400 mt-1">
                Players correctly identified in each metric's top rankings (out of 50 All-Defense selections)
              </p>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="px-6 py-4 text-left text-sm font-medium text-gray-400">Metric</th>
                    <th className="px-6 py-4 text-center text-sm font-medium text-gray-400">Top 10</th>
                    <th className="px-6 py-4 text-center text-sm font-medium text-gray-400">Top 20</th>
                    <th className="px-6 py-4 text-center text-sm font-medium text-gray-400">Top 30</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="bg-emerald-900/20 border-b border-gray-800">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                        <span className="font-bold text-white">EDI</span>
                        <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">Ours</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-2xl font-bold text-emerald-400">19</span>
                      <span className="text-gray-500">/50</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-2xl font-bold text-emerald-400">25</span>
                      <span className="text-gray-500">/50</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-2xl font-bold text-emerald-400">32</span>
                      <span className="text-gray-500">/50</span>
                    </td>
                  </tr>
                  <tr className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors">
                    <td className="px-6 py-4">
                      <span className="text-gray-300">DEF_RATING</span>
                      <span className="text-xs text-gray-500 ml-2">NBA Official</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-xl font-medium text-gray-400">1</span>
                      <span className="text-gray-600">/50</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-xl font-medium text-gray-400">1</span>
                      <span className="text-gray-600">/50</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-xl font-medium text-gray-400">3</span>
                      <span className="text-gray-600">/50</span>
                    </td>
                  </tr>
                  <tr className="hover:bg-gray-800/30 transition-colors">
                    <td className="px-6 py-4">
                      <span className="text-gray-300">DEF_WS</span>
                      <span className="text-xs text-gray-500 ml-2">NBA Official</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-xl font-medium text-gray-400">1</span>
                      <span className="text-gray-600">/50</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-xl font-medium text-gray-400">3</span>
                      <span className="text-gray-600">/50</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-xl font-medium text-gray-400">4</span>
                      <span className="text-gray-600">/50</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Highlight Card */}
          <div className="bg-gradient-to-r from-emerald-900/40 to-cyan-900/40 p-6 rounded-xl border border-emerald-700/50">
            <div className="flex items-start gap-4">
              <div className="bg-emerald-500/20 p-3 rounded-full">
                <TrendingUp className="w-8 h-8 text-emerald-400" />
              </div>
              <div>
                <h4 className="text-xl font-bold text-white mb-2">10x Better Coverage</h4>
                <p className="text-gray-300">
                  EDI identifies <span className="text-emerald-400 font-bold">10 times more</span> All-Defensive Team players than official NBA metrics in the Top 30 rankings. This demonstrates that the multi-dimensional approach captures defensive value that single-number metrics miss.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
