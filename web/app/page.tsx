import { getPlayers, getSeasons, getMeta, getSeasonInfo } from '@/lib/data';
import Dashboard from '@/components/Dashboard';
import { Shield } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  const seasons = getSeasons();
  const currentSeason = seasons[0];
  const meta = getMeta();
  
  const allData: Record<string, any[]> = {};
  const seasonInfoMap: Record<string, { max_games_played: number }> = {};
  for (const season of seasons) {
    allData[season] = getPlayers(season);
    const info = getSeasonInfo(season);
    if (info) {
      seasonInfoMap[season] = { max_games_played: info.max_games_played };
    }
  }

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-10 h-10 text-emerald-500" />
              <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                NBA Defensive Index
              </h1>
            </div>
            <Link 
              href="/methodology" 
              className="text-sm text-gray-400 hover:text-emerald-400 transition-colors border border-gray-800 rounded-full px-4 py-1.5 hover:border-emerald-500/50"
            >
              Methodology
            </Link>
          </div>
          <p className="text-gray-400 max-w-2xl">
            Advanced defensive metrics powered by a 5-dimensional Bayesian model. 
            Evaluating Shot Suppression, Profile, Hustle, IQ, and Anchor capabilities.
          </p>
        </header>

        <Dashboard 
          initialSeason={currentSeason} 
          seasons={seasons} 
          allData={allData}
          generatedAt={meta.generated_at}
          seasonInfoMap={seasonInfoMap}
        />
      </div>
    </main>
  );
}
