const apiTarget =
  process.env.API_PROXY_TARGET || process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${apiTarget}/:path*` },
    ];
  },
};

export default nextConfig;
