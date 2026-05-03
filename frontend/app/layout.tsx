import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RPS Classifier",
  description: "Rock Paper Scissors Image Recognition",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-full antialiased">{children}</body>
    </html>
  );
}
