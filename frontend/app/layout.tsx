import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AIgnition · Forecasting",
  description: "Probabilistic revenue & ROAS forecasting for e-commerce marketing",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
