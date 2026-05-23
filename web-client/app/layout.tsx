import { cookies } from "next/headers";
import type { Metadata } from "next";

import "./globals.css";
import { Providers } from "@/components/shell/Providers";
import { COLOR_OPTIONS, getColorOption } from "@/lib/theme/tokens";

export const metadata: Metadata = {
  title: "AutoPrice — Client Preview",
  description: "Agentic price-and-promo optimisation for CPG.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const cookieStore = cookies();
  const option = getColorOption(cookieStore.get("ap_color")?.value);
  const initialMode = COLOR_OPTIONS[option].defaultMode;

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Inter+Tight:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
        />
      </head>
      <body className="min-h-screen bg-background text-foreground antialiased">
        <Providers initialOption={option} initialMode={initialMode}>
          {children}
        </Providers>
      </body>
    </html>
  );
}
