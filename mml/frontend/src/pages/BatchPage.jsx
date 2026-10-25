import { useState, useEffect } from 'react';
import axios from 'axios';
import TopHeader from '../components/TopHeader';
import { formatINR } from '../utils/currency';

export default function BatchPage() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filterRisk, setFilterRisk] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');

  const fetchDemoBatch = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get('/api/demo-batch');
      const batchRecords = res.data?.records || res.data?.data || (Array.isArray(res.data) ? res.data : []);
      setData(batchRecords);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to load demo batch');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDemoBatch();
  }, []);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    const reader = new FileReader();
    reader.onload = async (evt) => {
      try {
        const text = evt.target.result;
        // Parse CSV or JSON
        let records = [];
        if (file.name.endsWith('.json')) {
          records = JSON.parse(text);
        } else {
          // Simple CSV parser
          const lines = text.trim().split(/\r?\n/);
          if (lines.length < 2) throw new Error('CSV must contain a header and at least one data row.');
          const headers = lines[0].split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''));
          for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            const values = lines[i].split(',').map((v) => v.trim().replace(/^["']|["']$/g, ''));
            const record = {};
            headers.forEach((h, idx) => {
              record[h] = values[idx];
            });
            records.push(record);
          }
        }

        const res = await axios.post('/api/predict/batch', records);
        const scored = res.data?.records || res.data?.data || [];
        setData(scored);
      } catch (err) {
        setError(err.response?.data?.error || err.message || 'Failed to parse and score batch file.');
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  };

  const total = data.length;
  const churners = data.filter((d) => d.prediction === 1 || d.risk_level === 'High Risk').length;
  const churnRate = total > 0 ? ((churners / total) * 100).toFixed(1) : 0;
  const totalRevenueAtRiskUSD = data
    .filter((d) => d.prediction === 1 || d.risk_level === 'High Risk')
    .reduce((sum, d) => sum + (parseFloat(d.MonthlyCharges) || 0) * 12, 0);

  const estimatedSavedUSD = totalRevenueAtRiskUSD * 0.40;

  // Filtered rows
  const filteredData = data.filter((row) => {
    const isHigh = row.prediction === 1 || row.risk_level === 'High Risk';
    if (filterRisk === 'HIGH' && !isHigh) return false;
    if (filterRisk === 'LOW' && isHigh) return false;
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      const id = (row.customerID || '').toLowerCase();
      const contract = (row.Contract || '').toLowerCase();
      const internet = (row.InternetService || '').toLowerCase();
      return id.includes(term) || contract.includes(term) || internet.includes(term);
    }
    return true;
  });

  return (
    <>
      <TopHeader
        title="PORTFOLIO BATCH INFERENCE"
        subtitle="Multi-Customer Cohort Analysis & Risk Scoring (Rupees INR)"
      />

      <div className="kpi-grid">
        <div className="kpi-card blue">
          <div className="kpi-label">Cohort Size</div>
          <div className="kpi-value">{total}</div>
          <div className="kpi-desc">Total customers scored</div>
        </div>
        <div className="kpi-card red">
          <div className="kpi-label">Predicted Churners</div>
          <div className="kpi-value red">{churners}</div>
          <div className="kpi-desc">{churnRate}% of cohort at risk</div>
        </div>
        <div className="kpi-card green">
          <div className="kpi-label">Retained Customers</div>
          <div className="kpi-value green">{total - churners}</div>
          <div className="kpi-desc">{(100 - churnRate).toFixed(1)}% safe</div>
        </div>
        <div className="kpi-card gold">
          <div className="kpi-label">Annual Revenue At Risk</div>
          <div className="kpi-value gold">
            {formatINR(totalRevenueAtRiskUSD)}
          </div>
          <div className="kpi-desc">MRR risk in Indian Rupees</div>
        </div>
      </div>

      <div className="glass-card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header" style={{ flexWrap: 'wrap', gap: '0.75rem' }}>
          <div className="card-title">
            <span>📋</span> Cohort Records & Risk Distribution
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <label
              className="more-info-btn"
              style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem', cursor: 'pointer' }}
            >
              📤 Upload CSV / JSON
              <input
                type="file"
                accept=".csv,.json"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </label>
            <button
              onClick={fetchDemoBatch}
              disabled={loading}
              className="more-info-btn"
              style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem' }}
            >
              {loading ? 'Analyzing...' : '🔄 Reload Demo Batch'}
            </button>
          </div>
        </div>

        {/* Filters and search */}
        <div style={{ display: 'flex', gap: '1rem', margin: '0.75rem 0 1rem 0', flexWrap: 'wrap', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="Search by ID, Contract, Internet..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="form-input"
            style={{ maxWidth: '280px', padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
          />
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            {['ALL', 'HIGH', 'LOW'].map((risk) => (
              <button
                key={risk}
                onClick={() => setFilterRisk(risk)}
                style={{
                  padding: '0.35rem 0.75rem',
                  fontSize: '0.75rem',
                  borderRadius: '6px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  border: filterRisk === risk ? '1px solid #4a90d9' : '1px solid rgba(148,163,184,0.3)',
                  background: filterRisk === risk ? 'rgba(74,144,217,0.15)' : 'transparent',
                  color: filterRisk === risk ? '#4a90d9' : '#94a3b8'
                }}
              >
                {risk === 'ALL' ? 'All Customers' : risk === 'HIGH' ? '⚠️ High Risk' : '✅ Low Risk'}
              </button>
            ))}
          </div>
          <div style={{ marginLeft: 'auto', fontSize: '0.8rem', color: '#64748b' }}>
            Showing {filteredData.length} of {total} records | Potential Savings: <strong style={{ color: '#34d399' }}>{formatINR(estimatedSavedUSD)}</strong>
          </div>
        </div>

        {error && (
          <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444', borderRadius: '8px', color: '#f87171', fontSize: '0.85rem', marginBottom: '1rem' }}>
            ⚠️ {error}
          </div>
        )}

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(74,144,217,0.2)', color: '#94a3b8' }}>
                <th style={{ padding: '0.75rem' }}>Customer ID</th>
                <th style={{ padding: '0.75rem' }}>Contract</th>
                <th style={{ padding: '0.75rem' }}>Tenure</th>
                <th style={{ padding: '0.75rem' }}>Internet</th>
                <th style={{ padding: '0.75rem' }}>Monthly (₹)</th>
                <th style={{ padding: '0.75rem' }}>Churn Prob</th>
                <th style={{ padding: '0.75rem' }}>Risk Level</th>
                <th style={{ padding: '0.75rem' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.length === 0 ? (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                    {loading ? 'Executing ML pipeline inference...' : 'No customers matching current filters.'}
                  </td>
                </tr>
              ) : (
                filteredData.map((row, idx) => {
                  const isHigh = row.prediction === 1 || row.risk_level === 'High Risk';
                  const monthlyUSD = parseFloat(row.MonthlyCharges) || 0;
                  const probPct = row.churn_probability_pct ?? Math.round((row.Churn_Probability || 0) * 100);

                  return (
                    <tr
                      key={row.customerID || idx}
                      style={{
                        borderBottom: '1px solid rgba(74,144,217,0.08)',
                        backgroundColor: isHigh ? 'rgba(239,68,68,0.03)' : 'transparent'
                      }}
                    >
                      <td style={{ padding: '0.75rem', fontWeight: 600, color: '#0f172a' }}>
                        {row.customerID || `CUST-${idx + 1}`}
                      </td>
                      <td style={{ padding: '0.75rem', color: '#475569' }}>{row.Contract}</td>
                      <td style={{ padding: '0.75rem', color: '#475569' }}>{row.tenure} mo</td>
                      <td style={{ padding: '0.75rem', color: '#475569' }}>{row.InternetService}</td>
                      <td style={{ padding: '0.75rem', color: '#0f172a', fontWeight: 600 }}>
                        {formatINR(monthlyUSD)}
                      </td>
                      <td style={{ padding: '0.75rem', fontWeight: 700, color: isHigh ? '#f87171' : '#34d399' }}>
                        {probPct}%
                      </td>
                      <td style={{ padding: '0.75rem' }}>
                        <span style={{
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px',
                          fontSize: '0.7rem',
                          fontWeight: 600,
                          backgroundColor: isHigh ? 'rgba(239,68,68,0.2)' : 'rgba(52,211,153,0.2)',
                          color: isHigh ? '#f87171' : '#34d399'
                        }}>
                          {row.risk_level || (isHigh ? 'High Risk' : 'Low Risk')}
                        </span>
                      </td>
                      <td style={{ padding: '0.75rem', color: isHigh ? '#f87171' : '#34d399', fontWeight: 600 }}>
                        {isHigh ? '⚠️ At Risk' : '✅ Loyal'}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
