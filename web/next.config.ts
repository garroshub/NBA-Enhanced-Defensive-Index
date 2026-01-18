import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Base path for GitHub Pages (if not custom domain)
  // basePath: '/NBA', 
};

export default nextConfig;
