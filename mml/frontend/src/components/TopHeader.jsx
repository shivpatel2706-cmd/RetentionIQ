export default function TopHeader({ title, subtitle }) {
  return (
    <div className="top-header">
      <div className="header-title-block">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="header-actions">
        <div className="search-bar">
          <span className="search-icon">🔍</span>
          <input type="text" placeholder="Search Here" />
        </div>
        <div className="header-icon-btn" title="Notifications">✉️</div>
        <div className="header-icon-btn" title="Alerts">🔔</div>
        <div className="header-icon-btn" title="Settings">⚙️</div>
      </div>
    </div>
  );
}
