import UploadPanel from "@/components/UploadPanel";
import ForecastDashboard from "@/components/ForecastDashboard";
import BudgetSimulator from "@/components/BudgetSimulator";
import InsightsPanel from "@/components/InsightsPanel";

export default function Home() {
  return (
    <main className="mx-auto max-w-5xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-semibold">
          AIgnition · Revenue &amp; ROAS forecasting
        </h1>
        <p className="text-sm text-gray-500">
          Probabilistic forecasts with budget simulation and AI-assisted causal insight.
        </p>
      </header>
      <UploadPanel />
      <ForecastDashboard />
      <BudgetSimulator />
      <InsightsPanel />
    </main>
  );
}
