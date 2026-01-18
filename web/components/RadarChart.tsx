'use client';

import { ResponsiveContainer, RadarChart as RechartsRadar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Tooltip } from 'recharts';
import { PlayerScores } from '@/lib/types';

interface RadarChartProps {
  scores: PlayerScores;
  color?: string;
}

export default function RadarChart({ scores, color = '#10B981' }: RadarChartProps) {
  const data = [
    { subject: 'Suppression', A: scores.d1, fullMark: 100 },
    { subject: 'Profile', A: scores.d2, fullMark: 100 },
    { subject: 'Hustle', A: scores.d3, fullMark: 100 },
    { subject: 'IQ', A: scores.d4, fullMark: 100 },
    { subject: 'Anchor', A: scores.d5, fullMark: 100 },
  ];

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsRadar cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ fill: '#9CA3AF', fontSize: 11 }} 
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar
            name="Score"
            dataKey="A"
            stroke={color}
            strokeWidth={2}
            fill={color}
            fillOpacity={0.4}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'rgba(17, 24, 39, 0.9)', 
              borderColor: '#374151', 
              color: '#F3F4F6',
              borderRadius: '0.5rem',
              backdropFilter: 'blur(4px)'
            }}
            itemStyle={{ color: '#F3F4F6' }}
          />
        </RechartsRadar>
      </ResponsiveContainer>
    </div>
  );
}
