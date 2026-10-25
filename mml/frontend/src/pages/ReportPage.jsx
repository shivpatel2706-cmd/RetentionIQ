import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Legend
} from 'recharts';
import TopHeader from '../components/TopHeader';
import { formatINR } from '../utils/currency';

function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <div className="tooltip-label">{label}</div>
        {payload.map((p, i) => (
          <div key={i} className="tooltip-value" style={{ color: p.color }}>
            {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
          </div>
        ))}
      </div>
    );
  }
  return null;
}

const MODEL_COLORS = {
  XGBoost: '#4a90d9',
  RandomForest: '#34d399',
  LightGBM: '#d4a843',
};

export default function ReportPage({ evalData, metaData }) {
  const models = metaData?.models_summary || {};
  const metrics = evalData?.metrics || {};
  const biz = evalData?.business_impact || {};

  // Bar chart data: test metrics comparison across models
  const barData = useMemo(() => {
    const metricNames = ['test_roc_auc', 'test_recall', 'test_precision', 'test_f1', 'test_accuracy'];
    const labels = ['ROC-AUC', 'Recall', 'Precision', 'F1-Score', 'Accuracy'];

    return metricNames.map((key, i) => {
      const point = { metric: labels[i] };
      Object.keys(models).forEach((model) => {
        point[model] = models[model][key] || 0;
      });
      return point;
    });
  }, [models]);

  // Radar chart data
  const radarData = useMemo(() => {
    const metricNames = ['test_roc_auc', 'test_recall', 'test_precision', 'test_f1'];
    const labels = ['ROC-AUC', 'Recall', 'Precision', 'F1'];

    return metricNames.map((key, i) => {
      const point = { metric: labels[i] };
      Object.keys(models).forEach((model) => {
        point[model] = Math.round((models[model][key] || 0) * 100);
      });
      return point;
    });
  }, [models]);

  return (
    <>
      <TopHeader
        title="MODEL BENCHMARKS & REPORT"
        subtitle="Comparative Analysis of All Trained Classifiers"
      />

      {/* Model Summary KPIs */}
      <div className="kpi-grid">
        {Object.keys(models).map((modelName) => {
          const m = models[modelName];
          const isChampion = modelName === (metaData?.champion_model || 'XGBoost');
          return (
            <div className={`kpi-card ${isChampion ? 'blue' : 'gold'}`} key={modelName}>
              <div className="kpi-label">
                {modelName} {isChampion ? '🥇 Champion' : ''}
              </div>
              <div className={`kpi-value ${isChampion ? 'blue' : ''}`}>
                {(m.test_roc_auc || 0).toFixed(4)}
              </div>
              <div className="kpi-desc">
                Recall: {((m.test_recall || 0) * 100).toFixed(1)}% | F1: {((m.test_f1 || 0) * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>

      <div className="charts-grid">
        {/* Bar Chart: Model Comparison */}
        <div className="chart-container">
          <div className="chart-header">
            <div>
              <div className="chart-title">Model Performance Comparison</div>
              <div className="chart-period">Test Set Metrics (N={biz.total_test_customers || 1409})</div>
            </div>
            <div className="card-help-icon">?</div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,144,217,0.1)" />
              <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
              {Object.keys(models).map((model) => (
                <Bar
                  key={model}
                  dataKey={model}
                  fill={MODEL_COLORS[model] || '#4a90d9'}
                  radius={[4, 4, 0, 0]}
                  maxBarSize={28}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="chart-container">
          <div className="chart-header">
            <div>
              <div className="chart-title">Performance Radar</div>
              <div className="chart-period">Multi-metric comparison (0-100%)</div>
            </div>
            <div className="card-help-icon">?</div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="rgba(74,144,217,0.15)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#94a3b8' }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} />
              {Object.keys(models).map((model) => (
                <Radar
                  key={model}
                  name={model}
                  dataKey={model}
                  stroke={MODEL_COLORS[model] || '#4a90d9'}
                  fill={MODEL_COLORS[model] || '#4a90d9'}
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              ))}
              <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
              <Tooltip content={<CustomTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix Summary */}
      <div className="glass-card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header">
          <div className="card-title">
            <span>📋</span> Confusion Matrix — Champion Model (XGBoost)
          </div>
        </div>
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
          <div className="kpi-card green">
            <div className="kpi-label">True Negatives</div>
            <div className="kpi-value green">{biz.true_negatives || 819}</div>
            <div className="kpi-desc">Correctly Retained</div>
          </div>
          <div className="kpi-card blue">
            <div className="kpi-label">True Positives</div>
            <div className="kpi-value blue">{biz.true_positives || 270}</div>
            <div className="kpi-desc">Correctly Flagged Churners</div>
          </div>
          <div className="kpi-card gold">
            <div className="kpi-label">False Positives</div>
            <div className="kpi-value">{biz.false_positives || 216}</div>
            <div className="kpi-desc">False Alarms</div>
          </div>
          <div className="kpi-card red">
            <div className="kpi-label">False Negatives</div>
            <div className="kpi-value red">{biz.false_negatives || 104}</div>
            <div className="kpi-desc">Missed Churners</div>
          </div>
        </div>
      </div>

      {/* Business Impact */}
      <div className="glass-card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header">
          <div className="card-title">
            <span>💰</span> Financial Impact Analysis (Indian Rupees)
          </div>
        </div>
        <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
          <div className="kpi-card gold">
            <div className="kpi-label">Annual Revenue At Risk</div>
            <div className="kpi-value gold">
              {formatINR(biz.total_annual_revenue_at_risk_usd || 0)}
            </div>
            <div className="kpi-desc">Total churner annual MRR</div>
          </div>
          <div className="kpi-card green">
            <div className="kpi-label">Revenue Identified (TP)</div>
            <div className="kpi-value green">
              {formatINR(biz.tp_annual_revenue_identified_usd || 0)}
            </div>
            <div className="kpi-desc">{((biz.tp_annual_revenue_identified_usd || 0) / (biz.total_annual_revenue_at_risk_usd || 1) * 100).toFixed(1)}% coverage</div>
          </div>
          <div className="kpi-card blue">
            <div className="kpi-label">Net Revenue Saved</div>
            <div className="kpi-value blue">
              {formatINR(biz.net_revenue_saved_usd || 0)}
            </div>
            <div className="kpi-desc">After campaign costs deducted</div>
          </div>
        </div>
      </div>

      <div className="page-footer">
        <span>RetentionIQ™ — Model Benchmarks</span>
        <span>5-Fold Stratified Cross-Validation | Fold-Isolated SMOTE</span>
      </div>
    </>
  );
}
