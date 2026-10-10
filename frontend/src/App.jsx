
import Sidebar from "./components/layout/Sidebar";
import Header from "./components/layout/Header";
import "./App.css";

export default function App() {
  return (
    <div className="app-layout">
      <Sidebar />

      <main className="main-content">
        <Header />

        <section className="dashboard-placeholder">
          <div className="placeholder-icon">↗</div>
          <h2>Your pricing intelligence workspace</h2>
          <p>
            Causal analysis, pricing impact, and budget optimization
            will appear here.
          </p>
          <span className="placeholder-status">
            Dashboard foundation ready
          </span>
        </section>
      </main>
    </div>
  );
}
