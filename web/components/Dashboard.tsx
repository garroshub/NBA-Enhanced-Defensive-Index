'use client';

import { useState } from 'react';
import { Player } from '@/lib/types';
import PlayerTable from './PlayerTable';
import { Calendar, Clock, Gamepad2 } from 'lucide-react';

interface DashboardProps {
  initialSeason: string;
  seasons: string[];
  allData: Record<string, Player[]>;
  generatedAt: string;
  seasonInfoMap: Record<string, { max_games_played: number }>;
}

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}

export default function Dashboard({ initialSeason, seasons, allData, generatedAt, seasonInfoMap }: DashboardProps) {
  const [currentSeason, setCurrentSeason] = useState(initialSeason);
  const players = allData[currentSeason] || [];

  const isCurrentSeason = currentSeason === seasons[0];
  
  // Calculate min games requirement: max_games_played // 2
  const seasonInfo = seasonInfoMap[currentSeason];
  const minGames = seasonInfo ? Math.floor(seasonInfo.max_games_played / 2) : null;

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
            {isCurrentSeason ? 'In Progress' : 'Final'}
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

      <PlayerTable players={players} season={currentSeason} isCurrentSeason={isCurrentSeason} />
    </div>
  );
}
