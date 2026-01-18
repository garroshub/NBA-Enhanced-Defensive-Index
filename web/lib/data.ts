import { promises as fs } from 'fs';
import path from 'path';
import { DataFile, Player } from './types';

export async function getData(): Promise<DataFile> {
  // This works on server side (build time)
  const filePath = path.join(process.cwd(), 'public', 'data.json');
  const fileContents = await fs.readFile(filePath, 'utf8');
  return JSON.parse(fileContents);
}

export async function getSeasons(): Promise<string[]> {
  const data = await getData();
  return Object.keys(data.seasons).sort().reverse();
}

export async function getPlayers(season?: string): Promise<Player[]> {
  const data = await getData();
  const seasons = Object.keys(data.seasons).sort().reverse();
  const targetSeason = season || seasons[0];
  return data.seasons[targetSeason] || [];
}

export async function getPlayerById(id: number, season: string = '2025-26'): Promise<Player | undefined> {
  const players = await getPlayers(season);
  return players.find(p => p.id === id);
}
