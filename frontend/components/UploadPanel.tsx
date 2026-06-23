"use client";
import { useState } from "react";
import { uploadFiles, validate } from "@/lib/api";

type Report = {
  ok: boolean;
  n_rows: number;
  n_campaigns: number;
  date_min: string;
  date_max: string;
  issues: string[];
};

export default function UploadPanel() {
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files) return;
    setBusy(true);
    try {
      await uploadFiles(Array.from(e.target.files));
      setReport(await validate());
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-lg border bg-white p-4">
      <h2 className="text-lg font-medium">1 · Upload data</h2>
      <input
        type="file"
        multiple
        accept=".csv"
        onChange={onUpload}
        className="mt-2 text-sm"
      />
      {busy && <p className="text-sm text-gray-500">Validating…</p>}
      {report && (
        <div className="mt-3 text-sm">
          <p>
            Rows: {report.n_rows} · Campaigns: {report.n_campaigns}
          </p>
          <p>
            Dates: {report.date_min} → {report.date_max}
          </p>
          <p className={report.ok ? "text-green-700" : "text-red-700"}>
            {report.ok ? "Validation passed" : `Issues: ${report.issues.join("; ")}`}
          </p>
        </div>
      )}
    </section>
  );
}
