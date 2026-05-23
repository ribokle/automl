const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const usdCompact = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 1,
});

const intCompact = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const intFull = new Intl.NumberFormat("en-US");

export const fmtUsd = (n: number) => usd.format(n);
export const fmtUsdCompact = (n: number) => usdCompact.format(n);
export const fmtIntCompact = (n: number) => intCompact.format(n);
export const fmtInt = (n: number) => intFull.format(n);
export const fmtPct = (n: number, digits = 1) =>
  `${n > 0 ? "+" : ""}${(n * 100).toFixed(digits)}%`;
export const fmtPctRaw = (n: number, digits = 1) =>
  `${(n * 100).toFixed(digits)}%`;
