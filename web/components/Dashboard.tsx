'use client';

import { useState } from 'react';
import { Player } from '@/lib/types';
import PlayerTable from './PlayerTable';
import { Calendar } from 'lucide-react';

interface DashboardProps {
  initialSeason: string;
  seasons: string[];
  allData: Record<string, Player[]>;
}

export default function Dashboard({ initialSeason, seasons, allData }: DashboardProps) {
  const [currentSeason, setCurrentSeason] = useState(initialSeason);
  const players = allData[currentSeason] || [];

  const isCurrentSeason = currentSeason === seasons[0];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
          <div className="text-sm text-gray-500">Status</div>
          <div className="text-xl font-semibold text-white">
            {isCurrentSeason ? 'In Progress' : 'Final'}
          </div>
        </div>
      </div>

      <PlayerTable players={players} season={currentSeason} isCurrentSeason={isCurrentSeason} />
    </div>
  );
}
