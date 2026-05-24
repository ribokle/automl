/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const clientUrl = process.env.BUSINESS_CLIENT_URL ?? "http://localhost:3100";
    return [
      { source: "/business", destination: `${clientUrl}/business` },
      { source: "/business/:path*", destination: `${clientUrl}/business/:path*` },
    ];
  },
};

export default nextConfig;
