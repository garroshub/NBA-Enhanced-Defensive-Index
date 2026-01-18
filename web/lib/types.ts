export interface PlayerScores {
  edi: number;
  d1: number;
  d2: number;
  d3: number;
  d4: number;
  d5: number;
  efficiency: number;
}

export interface PlayerRanks {
  overall: number;
}

export interface PlayerStats {
  gp: number;
  min: number;
  stl: number;
  blk: number;
}

export interface PlayerTrend {
  edi_change: number | null;
  rank_change: number | null;
  status: 'up' | 'down' | 'same' | 'new';
}

export interface Player {
  id: number;
  name: string;
  team: string | null;
  position: string | null;
  role: string;
  scores: PlayerScores;
  ranks: PlayerRanks;
  stats: PlayerStats;
  confidence: 'high' | 'medium' | 'low';
  trend: PlayerTrend;
}

export interface SeasonMetadata {
  total_players: number;
  max_games_played: number;
  season_progress: number;
  dynamic_c: number;
  is_current: boolean;
  generated_at?: string;
}

export interface SeasonData {
  [season: string]: Player[];
}

export interface DataFile {
  metadata: SeasonMetadata; // Note: The current JSON structure might have metadata inside seasons or separate. Let's check.
  seasons: SeasonData;
}
