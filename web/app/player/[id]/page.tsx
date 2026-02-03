import { getPlayers, getSeasons } from '@/lib/data';
import PlayerDetail from '@/components/PlayerDetail';
import { Suspense } from 'react';

// Generate static params for all players across all seasons
export function generateStaticParams() {
  try {
    const seasons = getSeasons();
    const allPlayerIds = new Set<string>();

    for (const season of seasons) {
      const players = getPlayers(season);
      players.forEach(p => allPlayerIds.add(p.id.toString()));
    }

    const params = Array.from(allPlayerIds).map((id) => ({
      id: id,
    }));

    // If no data is available (e.g. during initial build before data generation),
    // return a dummy path to satisfy Next.js 'output: export' requirement.
    // This page will render the "Player Not Found" state.
    if (params.length === 0) {
      console.warn('No player data found during build. Generating fallback path.');
      return [{ id: '0' }];
    }

    return params;
  } catch (error) {
    console.error('Error in generateStaticParams:', error);
    // Fallback on error as well
    return [{ id: '0' }];
  }
}

function LoadingFallback() {
  return (
    <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
      <div className="text-center">
        <div className="text-2xl font-bold mb-4">Loading...</div>
      </div>
    </div>
  );
}

export default async function PlayerPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = await params;
  return (
    <Suspense fallback={<LoadingFallback />}>
      <PlayerDetail id={parseInt(resolvedParams.id)} />
    </Suspense>
  );
}
