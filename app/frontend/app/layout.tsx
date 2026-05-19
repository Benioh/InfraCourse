import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Suspense } from "react";
import { AppShell } from "@/components/ui";
import AiTutor from "@/components/ai-tutor";
import "./globals.css";

export const metadata: Metadata = {
  title: "Infra Quest 中文课程",
  description: "多模态大模型 Infra 工程训练系统",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>
        <AppShell>{children}</AppShell>
        <Suspense fallback={null}>
          <AiTutor />
        </Suspense>
      </body>
    </html>
  );
}
