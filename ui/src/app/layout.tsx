import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/Navbar";
import { QueryProvider } from "@/providers/query-provider";

export const metadata: Metadata = {
  title: "Book Reco",
  description: "Book Recommendation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://rsms.me/" />
        <link rel="stylesheet" href="https://rsms.me/inter/inter.css" />
      </head>
      <body className="antialiased">
        <QueryProvider>
          <Navbar />
          {children}
        </QueryProvider>
      </body>
    </html>
  );
}
