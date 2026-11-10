import { useState } from "react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import Sidebar from "./components/layout/Sidebar";
import Header from "./components/layout/Header";

// Pages
import OverviewDashboard from "./pages/OverviewDashboard";
import DatasetsPage from "./pages/DatasetsPage";
import TrainingPage from "./pages/TrainingPage";
import CausalPage from "./pages/CausalPage";
import UpliftPage from "./pages/UpliftPage";
import PricingPage from "./pages/PricingPage";
import PromotionsPage from "./pages/PromotionsPage";
import SettingsPage from "./pages/SettingsPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";

import "./App.css";

function AppMain() {
  const { isAuthenticated, isLoading, role } = useAuth();
  const [activePage, setActivePage] = useState("Overview");
  const [authView, setAuthView] = useState("login");

  if (isLoading) {
    return (
      <div className="auth-container">
        <div className="auth-card" style={{ textAlign: "center" }}>
          <h2>Initializing EconoCausal...</h2>
          <p style={{ color: "#64748b" }}>Connecting to analytics workspace</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    if (authView === "register") {
      return <RegisterPage onSwitchToLogin={() => setAuthView("login")} />;
    }
    return <LoginPage onSwitchToRegister={() => setAuthView("register")} />;
  }

  const userRole = (role || "viewer").toLowerCase();

  const renderPage = () => {
    switch (activePage) {
      case "Overview":
        return <OverviewDashboard />;

      case "Datasets":
        if (userRole === "viewer") return <AccessRestricted page="Data Management" />;
        return <DatasetsPage />;

      case "Training":
        if (userRole === "viewer") return <AccessRestricted page="Model Training" />;
        return <TrainingPage />;

      case "Causal Effects":
        return <CausalPage />;

      case "Uplift Analysis":
        return <UpliftPage />;

      case "Pricing":
        return <PricingPage />;

      case "Promotions":
        if (userRole === "viewer") return <AccessRestricted page="Promotion Optimization" />;
        return <PromotionsPage />;

      case "Settings":
        return <SettingsPage />;

      default:
        return (
          <section className="dashboard-placeholder">
            <h2>{activePage}</h2>
            <p>This section is not yet available.</p>
          </section>
        );
    }
  };

  return (
    <div className="app-layout">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <main className="main-content">
        <Header />
        {renderPage()}
      </main>
    </div>
  );
}

function AccessRestricted({ page }) {
  return (
    <section className="dashboard-placeholder">
      <h2>Access Restricted</h2>
      <p>Your Viewer account does not have permission to access {page}.</p>
    </section>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppMain />
    </AuthProvider>
  );
}
