import { useState, useEffect } from 'react';
import axios from 'axios';
import TopHeader from '../components/TopHeader';
import { formatINR, formatINRDirect, USD_TO_INR } from '../utils/currency';

const BASELINE = {
  gender: 'Female',
  SeniorCitizen: 0,
  Partner: 'No',
  Dependents: 'No',
  tenure: 6,
  PhoneService: 'Yes',
  MultipleLines: 'No',
  InternetService: 'Fiber optic',
  OnlineSecurity: 'No',
  OnlineBackup: 'No',
  DeviceProtection: 'No',
  TechSupport: 'No',
  StreamingTV: 'Yes',
  StreamingMovies: 'Yes',
  Contract: 'Month-to-month',
  PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check',
  MonthlyCharges: 90.0,
  TotalCharges: 540.0
};

export default function SimulatorPage() {
  const [scenarioContract, setScenarioContract] = useState('Two year');
  const [scenarioTechSupport, setScenarioTechSupport] = useState('Yes');
  const [scenarioDiscountINR, setScenarioDiscountINR] = useState(1000); // in Rupees (₹)
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const discountUSD = scenarioDiscountINR / USD_TO_INR;
      const modifiedChargesUSD = Math.max(18.5, BASELINE.MonthlyCharges - discountUSD);
      const res = await axios.post('/api/simulate', {
        baseline: BASELINE,
        intervention: {
          Contract: scenarioContract,
          TechSupport: scenarioTechSupport,
          OnlineSecurity: scenarioTechSupport === 'Yes' ? 'Yes' : 'No',
          MonthlyCharges: modifiedChargesUSD,
          TotalCharges: BASELINE.tenure * modifiedChargesUSD
        }
      });
      if (res.data?.status === 'success' || res.data?.baseline) {
        setResult(res.data);
      } else {
        setError(res.data?.error || 'Simulation failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Error running simulation');
    } finally {
      setLoading(false);
    }
  };

  // Run automatically once on mount
  useEffect(() => {
    runSimulation();
  }, []);

  return (
    <>
      <TopHeader
        title="WHAT-IF RETENTION SIMULATOR"
        subtitle="Simulate Interventions & Quantify Probability Drop (Rupees INR)"
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
        {/* Intervention controls */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title">
              <span>🛠️</span> Proactive Retention Levers
            </div>
          </div>
          <p style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '1.25rem', lineHeight: 1.5 }}>
            Simulate the effect of proactive retention offers (contract upgrades, priority tech support, bill discounts in Rupees) on an at-risk subscriber.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label className="form-label">Upgrade Contract Type</label>
              <select
                value={scenarioContract}
                onChange={(e) => setScenarioContract(e.target.value)}
                className="form-input"
              >
                <option value="Month-to-month">Month-to-month (No change)</option>
                <option value="One year">One year commitment</option>
                <option value="Two year">Two year contract (Recommended)</option>
              </select>
            </div>

            <div>
              <label className="form-label">Add Tech Support & Cyber Shield Bundle</label>
              <select
                value={scenarioTechSupport}
                onChange={(e) => setScenarioTechSupport(e.target.value)}
                className="form-input"
              >
                <option value="No">No (Keep unprotected)</option>
                <option value="Yes">Yes (Add VIP Tech Support & Shield)</option>
              </select>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label className="form-label">Monthly Retention Discount (₹)</label>
                <span style={{ fontWeight: 700, color: '#4a90d9', fontSize: '0.9rem' }}>
                  {formatINRDirect(scenarioDiscountINR)}/mo
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="3000"
                step="250"
                value={scenarioDiscountINR}
                onChange={(e) => setScenarioDiscountINR(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--neon-blue)', cursor: 'pointer', marginTop: '0.35rem' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#64748b' }}>
                <span>₹0 discount</span>
                <span>₹1,500 discount</span>
                <span>₹3,000 discount</span>
              </div>
            </div>

            <div style={{ background: '#f8fafc', padding: '0.75rem', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.78rem', color: '#475569' }}>
              <div><strong>Baseline Plan:</strong> Fiber optic, Month-to-month</div>
              <div><strong>Baseline Monthly Spend:</strong> {formatINR(BASELINE.MonthlyCharges)}/mo</div>
            </div>

            <button
              onClick={runSimulation}
              disabled={loading}
              className="more-info-btn"
              style={{
                width: '100%',
                marginTop: '0.5rem',
                justifyContent: 'center',
                padding: '0.8rem',
                fontSize: '0.9rem',
                fontWeight: 600
              }}
            >
              {loading ? 'Running ML Simulation...' : '🧪 Run ML Simulation'}
            </button>
          </div>

          {error && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444', borderRadius: '8px', color: '#f87171', fontSize: '0.85rem' }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title">
              <span>📊</span> Intervention Impact
            </div>
            {result && (
              <span className="status-badge">
                Impact: <strong>{result.impact_level?.toUpperCase()}</strong>
              </span>
            )}
          </div>

          {result ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'center' }}>
                <div style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '12px', padding: '1rem' }}>
                  <div style={{ fontSize: '0.75rem', color: '#dc2626', fontWeight: 600 }}>BASELINE CHURN PROB</div>
                  <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#dc2626', margin: '0.4rem 0' }}>
                    {result.baseline?.churn_probability_pct}%
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    Status: <strong style={{ color: '#dc2626' }}>{result.baseline?.risk_level}</strong>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.2rem' }}>
                    Spend: {formatINRDirect(result.baseline?.monthly_charges_inr || BASELINE.MonthlyCharges * USD_TO_INR)}/mo
                  </div>
                </div>

                <div style={{ background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '12px', padding: '1rem' }}>
                  <div style={{ fontSize: '0.75rem', color: '#059669', fontWeight: 600 }}>POST-INTERVENTION</div>
                  <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#059669', margin: '0.4rem 0' }}>
                    {result.simulated?.churn_probability_pct}%
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    Status: <strong style={{ color: '#059669' }}>{result.simulated?.risk_level}</strong>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.2rem' }}>
                    Adjusted: {formatINRDirect(result.simulated?.monthly_charges_inr || (BASELINE.MonthlyCharges * USD_TO_INR - scenarioDiscountINR))}/mo
                  </div>
                </div>
              </div>

              {/* Risk drop callout */}
              <div style={{ textAlign: 'center', background: 'rgba(74,144,217,0.08)', border: '1px solid rgba(74,144,217,0.25)', borderRadius: '12px', padding: '1rem' }}>
                <div style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: 600 }}>CHURN RISK REDUCTION</div>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#2563eb', margin: '0.25rem 0' }}>
                  {result.delta_pct > 0 ? `-${result.delta_pct}%` : `+${Math.abs(result.delta_pct)}%`}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#334155' }}>
                  {result.delta_pct > 0
                    ? '🎉 Retention levers successfully mitigate churn risk down to safe levels!'
                    : 'Risk unchanged or elevated.'}
                </div>
              </div>

              {/* Financial comparison */}
              <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '10px', border: '1px solid #e2e8f0', fontSize: '0.82rem' }}>
                <div style={{ fontWeight: 600, color: '#0f172a', marginBottom: '0.4rem' }}>
                  💰 Financial Retention Payoff (Annual):
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569', marginBottom: '0.25rem' }}>
                  <span>Annual Account Value Saved:</span>
                  <strong style={{ color: '#059669' }}>
                    {formatINRDirect(BASELINE.MonthlyCharges * 12 * USD_TO_INR)}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569', marginBottom: '0.25rem' }}>
                  <span>Annual Retention Concession Cost:</span>
                  <strong style={{ color: '#dc2626' }}>
                    -{formatINRDirect(scenarioDiscountINR * 12)}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #cbd5e1', paddingTop: '0.4rem', marginTop: '0.4rem', color: '#0f172a', fontWeight: 700 }}>
                  <span>Net Annual Revenue Preserved:</span>
                  <span style={{ color: '#2563eb' }}>
                    {formatINRDirect(Math.max(0, BASELINE.MonthlyCharges * 12 * USD_TO_INR - scenarioDiscountINR * 12))}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem', opacity: 0.5 }}>⚡</div>
              <div style={{ fontSize: '0.95rem', color: '#0f172a', fontWeight: 600 }}>Select Levers & Run Simulation</div>
              <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.4rem' }}>
                Compare baseline customer risk against post-intervention scenarios in real time.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
