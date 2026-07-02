import { AllDefensiveComparison, DataFile, Player } from './types';
import rawData from './data.json';

const data: DataFile = rawData as unknown as DataFile;

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

export function getMeta() {
  return data.meta;
}

export function getSeasonInfo(season: string) {
  const key = `${season}_info`;
  return data.meta[key] as {
    total_players: number;
    max_games_played: number;
    season_progress: number;
    dynamic_c: number;
    is_current: boolean;
    is_final?: boolean;
  } | undefined;
}

export function getAllDefensiveComparison(season: string): AllDefensiveComparison | undefined {
  const key = `${season}_all_defensive_comparison`;
  return data.meta[key] as AllDefensiveComparison | undefined;
}
