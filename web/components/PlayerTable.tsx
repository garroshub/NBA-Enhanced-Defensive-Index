'use client';

import { useState } from 'react';
import { Player, PlayerScores } from '@/lib/types';
import { ArrowUpDown, Search, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import Link from 'next/link';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface PlayerTableProps {
  players: Player[];
  season: string;
}

type SortField = 'edi' | 'd1' | 'd2' | 'd3' | 'd4' | 'd5' | 'efficiency' | 'gp';
type SortDirection = 'asc' | 'desc';

export default function PlayerTable({ players, season }: PlayerTableProps) {
  const [sortField, setSortField] = useState<SortField>('edi');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState<string>('all');
  const isCurrentSeason = season === '2025-26'; // TODO: Make this dynamic based on metadata

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const filteredPlayers = players.filter(player => {
    const matchesSearch = player.name.toLowerCase().includes(search.toLowerCase()) || 
                          (player.team && player.team.toLowerCase().includes(search.toLowerCase()));
    const matchesRole = roleFilter === 'all' || player.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  const sortedPlayers = [...filteredPlayers].sort((a, b) => {
    let valA: number;
    let valB: number;

    if (sortField === 'gp') {
      valA = a.stats.gp;
      valB = b.stats.gp;
    } else if (sortField === 'edi') {
      valA = a.scores.edi;
      valB = b.scores.edi;
    } else {
      valA = a.scores[sortField];
      valB = b.scores[sortField];
    }

    return sortDirection === 'asc' ? valA - valB : valB - valA;
  });

  const getTrendIcon = (trend: Player['trend'] | undefined) => {
    if (!trend || !trend.edi_change) return <Minus className="w-4 h-4 text-gray-500" />;
    if (trend.edi_change > 0) return <TrendingUp className="w-4 h-4 text-emerald-500" />;
    if (trend.edi_change < 0) return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <Minus className="w-4 h-4 text-gray-500" />;
  };

  return (
    <div className="w-full space-y-4">
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-center bg-gray-900/50 p-4 rounded-lg border border-gray-800 backdrop-blur-sm">
        <div className="relative w-full sm:w-72">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search player or team..."
            className="w-full bg-gray-800 border border-gray-700 rounded-md pl-10 pr-4 py-2 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="flex gap-2">
          {['all', 'Guards', 'Frontcourt'].map((role) => (
            <button
              key={role}
              onClick={() => setRoleFilter(role)}
              className={cn(
                "px-3 py-1.5 text-sm rounded-md transition-colors",
                roleFilter === role 
                  ? "bg-emerald-600 text-white" 
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              )}
            >
              {role === 'all' ? 'All Roles' : role}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-800 bg-gray-900/30">
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-gray-400 uppercase bg-gray-900/80 border-b border-gray-800">
            <tr>
              <th className="px-4 py-3 font-medium">Rank</th>
              <th className="px-4 py-3 font-medium">Player</th>
              <th 
                className="px-4 py-3 font-medium cursor-pointer hover:text-emerald-400 transition-colors"
                onClick={() => handleSort('edi')}
              >
                <div className="flex items-center gap-1">
                  EDI
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              {isCurrentSeason && (
                <th className="px-4 py-3 font-medium text-center">Trend</th>
              )}
              <th 
                className="px-4 py-3 font-medium cursor-pointer hover:text-emerald-400 transition-colors text-right"
                onClick={() => handleSort('d1')}
              >
                D1 (Supp)
              </th>
              <th 
                className="px-4 py-3 font-medium cursor-pointer hover:text-emerald-400 transition-colors text-right"
                onClick={() => handleSort('d2')}
              >
                D2 (Prof)
              </th>
              <th 
                className="px-4 py-3 font-medium cursor-pointer hover:text-emerald-400 transition-colors text-right"
                onClick={() => handleSort('d3')}
              >
                D3 (Hust)
              </th>
              <th 
                className="px-4 py-3 font-medium cursor-pointer hover:text-emerald-400 transition-colors text-right"
                onClick={() => handleSort('d4')}
              >
                D4 (IQ)
              </th>
              <th 
                className="px-4 py-3 font-medium cursor-pointer hover:text-emerald-400 transition-colors text-right"
                onClick={() => handleSort('d5')}
              >
                D5 (Anch)
              </th>
              <th 
                className="px-4 py-3 font-medium cursor-pointer hover:text-emerald-400 transition-colors text-right"
                onClick={() => handleSort('efficiency')}
              >
                Eff
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {sortedPlayers.map((player, idx) => (
              <tr 
                key={player.id} 
                className="hover:bg-gray-800/50 transition-colors group"
              >
                <td className="px-4 py-3 font-medium text-gray-500">
                  #{player.ranks.overall}
                </td>
                <td className="px-4 py-3">
                  <Link href={`/player/${player.id}?season=${season}`} className="flex flex-col">
                    <span className="font-medium text-gray-200 group-hover:text-emerald-400 transition-colors">
                      {player.name}
                    </span>
                    <span className="text-xs text-gray-500">
                      {player.team} • {player.position} • {player.stats.gp} GP
                    </span>
                  </Link>
                </td>
                <td className="px-4 py-3 font-bold text-emerald-400 text-base">
                  {player.scores.edi.toFixed(1)}
                </td>
                {isCurrentSeason && (
                  <td className="px-4 py-3 text-center">
                    <div className="flex justify-center items-center gap-1">
                      {getTrendIcon(player.trend)}
                      {player.trend?.edi_change && (
                        <span className={cn(
                          "text-xs",
                          player.trend.edi_change > 0 ? "text-emerald-500" : "text-red-500"
                        )}>
                          {Math.abs(player.trend.edi_change).toFixed(1)}
                        </span>
                      )}
                    </div>
                  </td>
                )}
                {['d1', 'd2', 'd3', 'd4', 'd5'].map((key) => {
                  const scoreKey = key as keyof PlayerScores;
                  return (
                  <td key={key} className="px-4 py-3 text-right">
                    <div className="flex flex-col items-end gap-1">
                      <span className="text-gray-300">
                        {player.scores[scoreKey].toFixed(1)}
                      </span>
                      <div className="w-16 h-1 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-emerald-500/50" 
                          style={{ width: `${player.scores[scoreKey]}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  );
                })}
                <td className="px-4 py-3 text-right font-medium text-gray-300">
                  {player.scores.efficiency.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-center text-xs text-gray-500 pt-2">
        Showing {sortedPlayers.length} players
      </div>
    </div>
  );
}
