import { getPlayerById, getPlayers, getSeasons } from '@/lib/data';
import RadarChart from '@/components/RadarChart';
import Link from 'next/link';
import { ArrowLeft, Shield, Activity, Target, Brain, Anchor, Zap } from 'lucide-react';

// Generate static params for all players across all seasons
export async function generateStaticParams() {
  const seasons = await getSeasons();
  const allPlayerIds = new Set<string>();

  for (const season of seasons) {
    const players = await getPlayers(season);
    players.forEach(p => allPlayerIds.add(p.id.toString()));
  }

  return Array.from(allPlayerIds).map((id) => ({
    id: id,
  }));
}

export default async function PlayerPage({ params }: { params: { id: string } }) {
  const player = await getPlayerById(parseInt(params.id));

  if (!player) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Player Not Found</h1>
          <Link href="/" className="text-emerald-400 hover:underline">
            Return Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        <Link 
          href="/" 
          className="inline-flex items-center text-gray-400 hover:text-emerald-400 transition-colors mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Rankings
        </Link>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Profile & Stats */}
          <div className="space-y-6">
            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 backdrop-blur-sm">
              <div className="flex items-start justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-white mb-1">{player.name}</h1>
                  <div className="flex items-center gap-2 text-gray-400">
                    <span className="font-semibold text-emerald-400">{player.team}</span>
                    <span>•</span>
                    <span>{player.position}</span>
                    <span>•</span>
                    <span>{player.role}</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-500 uppercase tracking-wider">Rank</div>
                  <div className="text-4xl font-bold text-white">#{player.ranks.overall}</div>
                </div>
              </div>

              <div className="mt-8 grid grid-cols-2 gap-4">
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 uppercase">EDI Score</div>
                  <div className="text-2xl font-bold text-emerald-400">{player.scores.edi.toFixed(1)}</div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 uppercase">Efficiency</div>
                  <div className="text-2xl font-bold text-blue-400">{player.scores.efficiency.toFixed(2)}</div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 uppercase">Games</div>
                  <div className="text-xl font-semibold text-white">{player.stats.gp}</div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 uppercase">Minutes</div>
                  <div className="text-xl font-semibold text-white">{player.stats.min.toFixed(1)}</div>
                </div>
              </div>
            </div>

            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 backdrop-blur-sm">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-emerald-500" />
                Component Scores
              </h3>
              <div className="space-y-4">
                {[
                  { label: 'Shot Suppression (D1)', value: player.scores.d1, icon: Shield, desc: 'Ability to lower opponent FG%' },
                  { label: 'Shot Profile (D2)', value: player.scores.d2, icon: Target, desc: 'Forcing difficult shots (Rim/3PT)' },
                  { label: 'Hustle (D3)', value: player.scores.d3, icon: Zap, desc: 'Deflections, charges, contests' },
                  { label: 'Defensive IQ (D4)', value: player.scores.d4, icon: Brain, desc: 'Stocks to foul ratio' },
                  { label: 'Anchor/Reb (D5)', value: player.scores.d5, icon: Anchor, desc: 'Rebounding and paint protection' },
                ].map((item) => (
                  <div key={item.label}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300 flex items-center gap-2">
                        <item.icon className="w-3 h-3 text-gray-500" />
                        {item.label}
                      </span>
                      <span className="font-mono text-emerald-400">{item.value.toFixed(1)}</span>
                    </div>
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400" 
                        style={{ width: `${item.value}%` }}
                      />
                    </div>
                    <p className="text-[10px] text-gray-600 mt-0.5">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Radar Chart */}
          <div className="lg:col-span-2">
            <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 backdrop-blur-sm h-full min-h-[400px] flex flex-col">
              <h3 className="text-lg font-semibold text-white mb-6">Defensive Profile</h3>
              <div className="flex-grow flex items-center justify-center">
                <div className="w-full max-w-lg aspect-square">
                  <RadarChart scores={player.scores} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
