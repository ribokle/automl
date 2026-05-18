import type { ReactNode } from "react";

export default function MockupLayout({ children }: { children: ReactNode }) {
  return <div className="mx-auto max-w-[1280px] px-6 py-6 text-slate-100">{children}</div>;
}
