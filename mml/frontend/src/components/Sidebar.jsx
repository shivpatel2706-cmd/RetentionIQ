import { useState, useEffect } from 'react';

const PAGES = [
  { id: 'home', label: 'Home', icon: '🏠' },
  { id: 'report', label: 'Report', icon: '📊' },
  { id: 'predict', label: 'Predict', icon: '🔮' },
  { id: 'batch', label: 'Batch Analytics', icon: '📂' },
  { id: 'simulator', label: 'What-If', icon: '🧪' },
];

export default function Sidebar({ activePage, onPageChange, modelInfo }) {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="brand-icon">📡</div>
        <div className="brand-text">
          <span className="brand-name">RetentionIQ™</span>
          <span className="brand-edition">Enterprise Edition</span>
        </div>
      </div>

      {/* Profile */}
      <div className="sidebar-profile">
        <div className="profile-avatar">👤</div>
        <span className="profile-name">Data Analyst</span>
        <div className="profile-stars">
          {[1, 2, 3, 4].map((s) => (
            <span key={s}>★</span>
          ))}
          <span style={{ opacity: 0.3 }}>★</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {PAGES.map((page) => (
          <div
            key={page.id}
            className={`nav-item ${activePage === page.id ? 'active' : ''}`}
            onClick={() => onPageChange(page.id)}
          >
            <span className="nav-icon">{page.icon}</span>
            <span>{page.label}</span>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-logout">
          <span>🚪</span>
          <span>Log Out</span>
        </div>
      </div>
    </aside>
  );
}
