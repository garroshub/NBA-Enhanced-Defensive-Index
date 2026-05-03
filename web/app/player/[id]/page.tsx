import { getPlayers, getSeasons } from '@/lib/data';
import PlayerDetail from '@/components/PlayerDetail';
import { Suspense } from 'react';

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

    if (params.length === 0) {
      console.warn('No player data found during build. Generating fallback path.');
      return [{ id: '0' }];
    }

    return params;
  } catch (error) {
    console.error('Error in generateStaticParams:', error);
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
