
import { useState } from "react";
import Sidebar from "./components/layout/Sidebar";
import Header from "./components/layout/Header";
import DataUpload from "./pages/DataUpload";
import AnalyticsDashboard from "./pages/AnalyticsDashboard";
import AllocationDashboard from "./pages/AllocationDashboard";
import "./App.css";

export default function App() {
  const [activePage, setActivePage] = useState("Overview");

  const renderPage = () => {
    switch (activePage) {
      case "Data Management":
        return <DataUpload />;

      case "Budget Optimizer":
        return <AllocationDashboard />;

      case "Overview":
      case "Causal Effects":
      case "Uplift Analysis":
      case "Pricing Analysis":
        return <AnalyticsDashboard />;

      default:
        return (
          <section className="dashboard-placeholder">
            <h2>{activePage}</h2>
            <p>This section is planned for a later development day.</p>
          </section>
        );
    }
  };

  return (
    <div className="app-layout">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
      />

      <main className="main-content">
        <Header />
        {renderPage()}
      </main>
    </div>
  );
}
