import { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts';
import TopHeader from '../components/TopHeader';
import { formatINR } from '../utils/currency';

// Days of week for the 7-card grid
const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

// Custom tooltip for area chart
function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <div className="tooltip-label">{label}</div>
        {payload.map((p, i) => (
          <div key={i} className="tooltip-value" style={{ color: p.color }}>
            {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
          </div>
        ))}
      </div>
    );
  }
  return null;
}

const DONUT_COLORS = ['#34d399', '#f87171', '#fbbf24', '#4a90d9'];

export default function HomePage({ evalData, metaData }) {
  const [dailyData, setDailyData] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);

  useEffect(() => {
    if (!evalData || !metaData) return;

    const biz = evalData.business_impact || {};
    const metrics = evalData.metrics || {};
    const models = metaData.models_summary || {};

    // Build daily-card data from model metrics & business data
    const totalRevenue = biz.total_annual_revenue_at_risk_usd || 326578.80;
    const dailyRevenue = totalRevenue / 7;
    const features = evalData.top_10_features || [];

    const daily = DAYS.map((day, i) => {
      const feature = features[i] || {};
      return {
        day,
        dayNum: String(i + 1).padStart(2, '0'),
        value: formatINR(dailyRevenue),
        info: feature.feature
          ? feature.feature.replace(/_/g, ' ').substring(0, 30)
          : 'Churn metric',
      };
    });
    setDailyData(daily);

    // Area chart: simulated 7-day predictions trend using model scores
    const modelNames = Object.keys(models);
    const areaData = DAYS.map((day, i) => {
      const point = { name: `Day ${String(i + 1).padStart(2, '0')}` };
      modelNames.forEach((model) => {
        const m = models[model];
        // Create variation around test scores for visual variety
        const base = (m.test_recall || 0.7) * 100 * (1 + (Math.sin(i * 1.2) * 0.15));
        point[model] = Math.round(base * 10) / 10;
      });
      // Also add a "Customers" metric
      point['At-Risk Customers'] = Math.round(
        (biz.actual_churners || 374) / 7 * (1 + Math.sin(i * 0.8) * 0.3)
      );
      return point;
    });
    setChartData(areaData);

    // Donut comparison: churners vs retained
    setComparisonData([
      { name: 'Retained (TN)', value: biz.true_negatives || 819 },
      { name: 'Churned (TP)', value: biz.true_positives || 270 },
      { name: 'False Positive', value: biz.false_positives || 216 },
      { name: 'Missed (FN)', value: biz.false_negatives || 104 },
    ]);
  }, [evalData, metaData]);

  const biz = evalData?.business_impact || {};
  const metrics = evalData?.metrics || {};

  return (
    <>
      <TopHeader
        title="DAILY REPORT CONCEPT"
        subtitle="Dashboard with Infographic for Business Intelligence"
      />

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card blue">
          <div className="kpi-label">Total Revenue At Risk (Annual)</div>
          <div className="kpi-value gold">
            {formatINR(biz.total_annual_revenue_at_risk_usd || 326578.80)}
          </div>
          <div className="kpi-desc">Revenue at risk in Indian Rupees</div>
        </div>
        <div className="kpi-card green">
          <div className="kpi-label">Revenue Preserved</div>
          <div className="kpi-value green">
            {formatINR(biz.net_revenue_saved_usd || 84135.36)}
          </div>
          <div className="kpi-desc">Net annual savings from retention</div>
        </div>
        <div className="kpi-card gold">
          <div className="kpi-label">Campaign ROI</div>
          <div className="kpi-value gold">
            +{(biz.roi_percentage || 692.47).toFixed(1)}%
          </div>
          <div className="kpi-desc">Return on outreach investment</div>
        </div>
        <div className="kpi-card red">
          <div className="kpi-label">Model Recall</div>
          <div className="kpi-value">
            {((metrics.recall || 0.7219) * 100).toFixed(1)}%
          </div>
          <div className="kpi-desc">Churner detection rate (XGBoost)</div>
        </div>
        <div className="kpi-card purple">
          <div className="kpi-label">ROC-AUC Score</div>
          <div className="kpi-value blue">
            {(metrics.roc_auc || 0.8439).toFixed(4)}
          </div>
          <div className="kpi-desc">Area under ROC curve</div>
        </div>
      </div>

      {/* 01. Report Section with 7-day grid */}
      <div className="report-section">
        <div className="report-summary">
          <div className="glass-card">
            <div className="section-number">01. Report</div>
            <div className="section-period">Periode: Day 01 - 07</div>
            <p className="summary-text">
              Comprehensive churn risk analysis across the telco subscriber portfolio.
              XGBoost champion model identifies at-risk customers with {((metrics.recall || 0.7219) * 100).toFixed(1)}% recall.
            </p>
            <div className="total-sales-block">
              <div className="total-sales-label">Total Revenue At Risk</div>
              <div className="total-sales-value">
                {formatINR(biz.total_annual_revenue_at_risk_usd || 326578.80)}
              </div>
            </div>
            <button className="more-info-btn">
              <span>●</span> more info
            </button>
          </div>
        </div>

        <div className="glass-card">
          <div className="daily-grid">
            {dailyData.map((d, i) => (
              <div className="daily-card" key={i}>
                <div className="day-number">{d.dayNum}</div>
                <div className="day-info">{d.info}</div>
                <div className="day-value">{d.value}</div>
                <div className="day-label">{d.day}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="charts-grid">
        {/* 02. Report Graph - Area Chart */}
        <div className="chart-container">
          <div className="chart-header">
            <div>
              <div className="chart-title">02. Report Graph</div>
              <div className="chart-period">Periode: Day 01 - 07</div>
            </div>
            <div className="card-help-icon">?</div>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="gradientBlue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4a90d9" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#4a90d9" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="gradientGold" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#d4a843" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#d4a843" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,144,217,0.1)" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="At-Risk Customers"
                stroke="#4a90d9"
                fill="url(#gradientBlue)"
                strokeWidth={2}
                dot={{ r: 3, fill: '#4a90d9' }}
              />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#4a90d9', display: 'inline-block' }}></span>
            <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>At-Risk Subscribers (Daily)</span>
          </div>
        </div>

        {/* 03. Comparison Report - Donut Chart */}
        <div className="chart-container">
          <div className="chart-header">
            <div>
              <div className="chart-title">03. Comparison Report</div>
            </div>
            <div className="card-help-icon">?</div>
          </div>
          <div className="comparison-container">
            <div className="comparison-stats">
              <div>
                <div className="comparison-item">
                  <span className="comparison-dot low"></span>
                  <span className="comparison-label">Retained (TN)</span>
                </div>
                <div className="comparison-value">{biz.true_negatives || 819}</div>
              </div>
              <div>
                <div className="comparison-item">
                  <span className="comparison-dot high"></span>
                  <span className="comparison-label">Churned Detected (TP)</span>
                </div>
                <div className="comparison-value">{biz.true_positives || 270}</div>
              </div>
              <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '0.5rem' }}>
                * Based on holdout test set (N={biz.total_test_customers || 1409})
              </div>
            </div>
            <div className="comparison-chart-wrapper">
              <ResponsiveContainer width={200} height={200}>
                <PieChart>
                  <Pie
                    data={comparisonData}
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={3}
                    dataKey="value"
                    stroke="none"
                  >
                    {comparisonData.map((entry, index) => (
                      <Cell key={index} fill={DONUT_COLORS[index % DONUT_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div style={{
            fontSize: '0.72rem',
            color: '#94a3b8',
            marginTop: '0.5rem',
            display: 'flex',
            gap: '1rem',
            flexWrap: 'wrap'
          }}>
            {comparisonData.map((d, i) => (
              <span key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: DONUT_COLORS[i], display: 'inline-block' }}></span>
                {d.name}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="glass-card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header">
          <div className="card-title">
            <span>🔑</span> Top 10 Churn Drivers
          </div>
          <div className="status-badge">
            <span className="pulse-dot"></span>
            <span>AI Engine Ready (XGBoost)</span>
          </div>
        </div>
        <div className="feature-bars">
          {(evalData?.top_10_features || []).map((f, i) => (
            <div className="feature-bar-item" key={i}>
              <span className="feature-bar-label">
                {f.feature.replace(/_/g, ' ').replace('cat__', '').replace('num__', '')}
              </span>
              <div className="feature-bar-track">
                <div
                  className="feature-bar-fill"
                  style={{ width: `${(f.importance_pct / 30) * 100}%` }}
                >
                  <span className="feature-bar-pct">{f.importance_pct.toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="page-footer">
        <span>RetentionIQ™ — Telco Churn Intelligence Platform</span>
        <span>Powered by XGBoost | {new Date().getFullYear()}</span>
      </div>
    </>
  );
}
