import { DataFile, Player } from './types';
import rawData from './data.json';

// Bundled data - imported at build time, guaranteed to exist
const data: DataFile = rawData as DataFile;

export function getData(): DataFile {
  return data;
}

export function getSeasons(): string[] {
  return Object.keys(data.seasons).sort().reverse();
}

export function getPlayers(season?: string): Player[] {
  const seasons = Object.keys(data.seasons).sort().reverse();
  const targetSeason = season || seasons[0];
  return data.seasons[targetSeason] || [];
}

export function getPlayerById(id: number, season: string = '2025-26'): Player | undefined {
  const players = getPlayers(season);
  return players.find(p => p.id === id);
}
