/**
 * KpiCard — Reusable metric card component.
 * Props: label, value, detail, accent (optional color)
 */
export function KpiCard({ label, value, detail, accent }) {
  return (
    <article className="kpi-card" style={accent ? { borderTop: `3px solid ${accent}` } : {}}>
      <span className="kpi-card__label">{label}</span>
      <span className="kpi-card__value">{value ?? "—"}</span>
      {detail && <span className="kpi-card__detail">{detail}</span>}
    </article>
  );
}

/**
 * StateView — Empty / loading / error placeholder.
 * Props: icon, title, desc, children (optional action)
 */
export function StateView({ icon = "○", title, desc, children }) {
  return (
    <div className="state-view">
      <div className="state-view__icon">{icon}</div>
      <p className="state-view__title">{title}</p>
      {desc && <p className="state-view__desc">{desc}</p>}
      {children}
    </div>
  );
}

/**
 * Badge — Status badge.
 * variant: success | warning | error | info | neutral
 */
export function Badge({ children, variant = "neutral" }) {
  return <span className={`badge badge-${variant}`}>{children}</span>;
}

/**
 * statusBadge — Maps a status string to a Badge variant.
 */
export function StatusBadge({ status }) {
  const map = {
    completed: "success",
    running: "info",
    pending: "warning",
    failed: "error",
    optimal: "success",
    infeasible: "error",
    cancelled: "neutral",
  };
  const variant = map[(status || "").toLowerCase()] || "neutral";
  return <Badge variant={variant}>{status}</Badge>;
}

/**
 * SectionHeader — Page-level section heading
 */
export function SectionHeader({ eyebrow, title, subtitle, children }) {
  return (
    <div className="page-header">
      <div>
        {eyebrow && <span className="page-header__eyebrow">{eyebrow}</span>}
        <h1 className="page-header__title">{title}</h1>
        {subtitle && <p className="page-header__subtitle">{subtitle}</p>}
      </div>
      {children && <div className="page-header__actions">{children}</div>}
    </div>
  );
}

/**
 * ErrorBanner — Displays an error message.
 */
export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="error-banner" role="alert">
      ⚠ {message}
    </div>
  );
}

/**
 * SuccessBanner — Displays a success message.
 */
export function SuccessBanner({ message }) {
  if (!message) return null;
  return (
    <div className="success-banner" role="status">
      ✓ {message}
    </div>
  );
}

