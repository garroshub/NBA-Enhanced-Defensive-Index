'use client';

import { useState } from 'react';
import { AllDefensiveComparison, Player } from '@/lib/types';
import PlayerTable from './PlayerTable';
import { Calendar, CheckCircle2, Clock, Gamepad2, Target, Trophy } from 'lucide-react';

interface DashboardProps {
  initialSeason: string;
  seasons: string[];
  allData: Record<string, Player[]>;
  generatedAt: string;
  seasonInfoMap: Record<string, { max_games_played: number; is_final?: boolean }>;
  allDefensiveComparisonMap: Record<string, AllDefensiveComparison>;
}

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}

export default function Dashboard({
  initialSeason,
  seasons,
  allData,
  generatedAt,
  seasonInfoMap,
  allDefensiveComparisonMap
}: DashboardProps) {
  const [currentSeason, setCurrentSeason] = useState(initialSeason);
  const players = allData[currentSeason] || [];

  const isLatestSeason = currentSeason === seasons[0];
  
  // Calculate min games requirement: max_games_played // 2
  const seasonInfo = seasonInfoMap[currentSeason];
  const minGames = seasonInfo ? Math.floor(seasonInfo.max_games_played / 2) : null;
  const isFinal = Boolean(seasonInfo?.is_final || (seasonInfo?.max_games_played ?? 0) >= 82);
  const comparison = allDefensiveComparisonMap[currentSeason];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-800">
          <div className="text-sm text-gray-500 mb-1 flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            Season
          </div>
          <select
            value={currentSeason}
            onChange={(e) => setCurrentSeason(e.target.value)}
            className="bg-gray-800 text-white border border-gray-700 rounded px-3 py-1 text-lg font-semibold w-full focus:ring-2 focus:ring-emerald-500 outline-none"
          >
            {seasons.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-800">
          <div className="text-sm text-gray-500">Players Analyzed</div>
          <div className="text-xl font-semibold text-white">{players.length}</div>
        </div>
        <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-800">
          <div className="text-sm text-gray-500 flex items-center gap-2">
            <Gamepad2 className="w-4 h-4" />
            Min Games
          </div>
          <div className="text-xl font-semibold text-white">
            {minGames !== null ? `${minGames} GP` : 'N/A'}
          </div>
        </div>
        <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-800">
          <div className="text-sm text-gray-500">Status</div>
          <div className="text-xl font-semibold text-white">
            {isFinal ? 'Final' : isLatestSeason ? 'In Progress' : 'Final'}
          </div>
        </div>
        <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-800">
          <div className="text-sm text-gray-500 flex items-center gap-2">
            <Clock className="w-4 h-4" />
            Last Updated
          </div>
          <div className="text-xl font-semibold text-white">{formatDate(generatedAt)}</div>
        </div>
      </div>

      {comparison && (
        <section className="space-y-4 rounded-lg border border-gray-800 bg-gray-900/40 p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-medium uppercase tracking-wide text-emerald-400">
                <Trophy className="h-4 w-4" />
                Official All-Defensive Comparison
              </div>
              <h2 className="mt-1 text-2xl font-semibold text-white">
                {comparison.season} EDI vs NBA All-Defensive Teams
              </h2>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="rounded-md border border-gray-800 bg-gray-950/70 px-3 py-2">
                <div className="text-xs text-gray-500">Top 10</div>
                <div className="text-lg font-semibold text-emerald-400">
                  {comparison.top_10_hits}/{comparison.official_count}
                </div>
              </div>
              <div className="rounded-md border border-gray-800 bg-gray-950/70 px-3 py-2">
                <div className="text-xs text-gray-500">Top 15</div>
                <div className="text-lg font-semibold text-cyan-300">
                  {comparison.top_15_hits}/{comparison.official_count}
                </div>
              </div>
              <div className="rounded-md border border-gray-800 bg-gray-950/70 px-3 py-2">
                <div className="text-xs text-gray-500">Top 30</div>
                <div className="text-lg font-semibold text-white">
                  {comparison.top_30_hits}/{comparison.official_count}
                </div>
              </div>
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="overflow-hidden rounded-lg border border-gray-800">
              <div className="flex items-center gap-2 border-b border-gray-800 bg-gray-950/80 px-4 py-3 text-sm font-medium text-gray-300">
                <Target className="h-4 w-4 text-emerald-400" />
                EDI Top 10
              </div>
              <div className="divide-y divide-gray-800">
                {comparison.top_10.map((player) => (
                  <div key={player.name} className="grid grid-cols-[48px_1fr_76px_120px] items-center gap-3 px-4 py-2 text-sm">
                    <span className="text-gray-500">#{player.rank}</span>
                    <span className="font-medium text-gray-100">{player.name}</span>
                    <span className="text-right font-semibold text-emerald-400">{player.edi.toFixed(1)}</span>
                    <span className={player.official_team ? 'text-cyan-300' : 'text-gray-600'}>
                      {player.official_team || 'Not selected'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="overflow-hidden rounded-lg border border-gray-800">
              <div className="flex items-center gap-2 border-b border-gray-800 bg-gray-950/80 px-4 py-3 text-sm font-medium text-gray-300">
                <CheckCircle2 className="h-4 w-4 text-cyan-300" />
                Official Selections by EDI Rank
              </div>
              <div className="divide-y divide-gray-800">
                {comparison.official_players.map((player) => (
                  <div key={player.name} className="grid grid-cols-[48px_1fr_76px] items-center gap-3 px-4 py-2 text-sm">
                    <span className="text-gray-500">#{player.edi_rank}</span>
                    <span>
                      <span className="font-medium text-gray-100">{player.name}</span>
                      <span className="ml-2 text-xs text-gray-500">{player.official_team}</span>
                    </span>
                    <span className="text-right font-semibold text-emerald-400">{player.edi.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      <PlayerTable players={players} season={currentSeason} />
    </div>
  );
}
