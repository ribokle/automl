/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  basePath: "/business",
  assetPrefix: "/business",
  experimental: {
    typedRoutes: false,
  },
};

export default nextConfig;
