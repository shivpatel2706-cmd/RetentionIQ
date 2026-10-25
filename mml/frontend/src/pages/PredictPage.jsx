import { useState } from 'react';
import axios from 'axios';
import TopHeader from '../components/TopHeader';
import { formatINR, USD_TO_INR } from '../utils/currency';

const DEFAULT_FORM = {
  gender: 'Female',
  SeniorCitizen: 0,
  Partner: 'Yes',
  Dependents: 'No',
  tenure: 12,
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
  MonthlyCharges: Math.round(85.5 * USD_TO_INR),
  TotalCharges: Math.round(1026.0 * USD_TO_INR)
};

export default function PredictPage() {
  const [formData, setFormData] = useState(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    let parsedValue = value;
    if (type === 'number') {
      parsedValue = parseFloat(value) || 0;
    }
    setFormData((prev) => {
      const updated = { ...prev, [name]: parsedValue };
      if (name === 'tenure' || name === 'MonthlyCharges') {
        const tenure = name === 'tenure' ? parsedValue : prev.tenure;
        const monthly = name === 'MonthlyCharges' ? parsedValue : prev.MonthlyCharges;
        updated.TotalCharges = parseFloat((tenure * monthly).toFixed(2));
      }
      return updated;
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post('/api/predict', formData);
      if (res.data.status === 'success') {
        setResult(res.data);
      } else {
        setError(res.data.error || 'Failed to generate prediction');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'API connection failed');
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (type) => {
    if (type === 'high_risk') {
      setFormData({
        gender: 'Female',
        SeniorCitizen: 1,
        Partner: 'No',
        Dependents: 'No',
        tenure: 3,
        PhoneService: 'Yes',
        MultipleLines: 'Yes',
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
        MonthlyCharges: Math.round(98.5 * USD_TO_INR),
        TotalCharges: Math.round(295.5 * USD_TO_INR)
      });
    } else {
      setFormData({
        gender: 'Male',
        SeniorCitizen: 0,
        Partner: 'Yes',
        Dependents: 'Yes',
        tenure: 58,
        PhoneService: 'Yes',
        MultipleLines: 'Yes',
        InternetService: 'DSL',
        OnlineSecurity: 'Yes',
        OnlineBackup: 'Yes',
        DeviceProtection: 'Yes',
        TechSupport: 'Yes',
        StreamingTV: 'No',
        StreamingMovies: 'No',
        Contract: 'Two year',
        PaperlessBilling: 'No',
        PaymentMethod: 'Bank transfer (automatic)',
        MonthlyCharges: Math.round(62.0 * USD_TO_INR),
        TotalCharges: Math.round(3596.0 * USD_TO_INR)
      });
    }
  };

  return (
    <>
      <TopHeader
        title="LIVE INFERENCE & PREDICTION"
        subtitle="Real-time XGBoost ML Scoring Engine"
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 'var(--space-xl)' }}>
        {/* Input Form */}
        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '1rem' }}>
            <div className="card-title">
              <span>👤</span> Customer Profile Inputs
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                type="button"
                className="more-info-btn"
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.72rem' }}
                onClick={() => loadPreset('high_risk')}
              >
                🔴 High Risk Profile
              </button>
              <button
                type="button"
                className="more-info-btn"
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.72rem' }}
                onClick={() => loadPreset('loyal')}
              >
                🟢 Loyal Profile
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.85rem' }}>
              {/* Demographics */}
              <div>
                <label className="form-label">Gender</label>
                <select name="gender" value={formData.gender} onChange={handleChange} className="form-input">
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                </select>
              </div>

              <div>
                <label className="form-label">Senior Citizen</label>
                <select name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleChange} className="form-input">
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div>
                <label className="form-label">Partner</label>
                <select name="Partner" value={formData.Partner} onChange={handleChange} className="form-input">
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="form-label">Dependents</label>
                <select name="Dependents" value={formData.Dependents} onChange={handleChange} className="form-input">
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              {/* Service Details */}
              <div>
                <label className="form-label">Tenure (Months)</label>
                <input
                  type="number"
                  name="tenure"
                  min="0"
                  max="72"
                  value={formData.tenure}
                  onChange={handleChange}
                  className="form-input"
                />
              </div>

              <div>
                <label className="form-label">Contract Type</label>
                <select name="Contract" value={formData.Contract} onChange={handleChange} className="form-input">
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              <div>
                <label className="form-label">Internet Service</label>
                <select name="InternetService" value={formData.InternetService} onChange={handleChange} className="form-input">
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="DSL">DSL</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="form-label">Payment Method</label>
                <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange} className="form-input">
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>

              <div>
                <label className="form-label">Online Security</label>
                <select name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleChange} className="form-input">
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="form-label">Tech Support</label>
                <select name="TechSupport" value={formData.TechSupport} onChange={handleChange} className="form-input">
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="form-label">Monthly Charges (₹)</label>
                <input
                  type="number"
                  step="1"
                  name="MonthlyCharges"
                  value={formData.MonthlyCharges}
                  onChange={handleChange}
                  className="form-input"
                />
              </div>

              <div>
                <label className="form-label">Total Charges (₹)</label>
                <input
                  type="number"
                  step="1"
                  name="TotalCharges"
                  value={formData.TotalCharges}
                  onChange={handleChange}
                  className="form-input"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="more-info-btn"
              style={{
                width: '100%',
                marginTop: '1.25rem',
                justifyContent: 'center',
                padding: '0.8rem',
                fontSize: '0.9rem',
                fontWeight: 600
              }}
            >
              {loading ? 'Evaluating Model...' : '⚡ Generate Churn Prediction'}
            </button>
          </form>

          {error && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444', borderRadius: '8px', color: '#f87171', fontSize: '0.85rem' }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Prediction Results Display */}
        <div>
          {result ? (
            <div className="glass-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <div className="card-header">
                <div className="card-title">
                  <span>🎯</span> ML Prediction Result
                </div>
                <div className="status-badge">
                  <span>Threshold: {result.threshold}</span>
                </div>
              </div>

              <div style={{ textAlign: 'center', margin: '1.5rem 0' }}>
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '140px',
                  height: '140px',
                  borderRadius: '50%',
                  border: `4px solid ${result.prediction === 1 ? '#ef4444' : '#34d399'}`,
                  background: result.prediction === 1 ? 'rgba(239,68,68,0.1)' : 'rgba(52,211,153,0.1)',
                  boxShadow: result.prediction === 1 ? '0 0 25px rgba(239,68,68,0.3)' : '0 0 25px rgba(52,211,153,0.3)',
                  flexDirection: 'column'
                }}>
                  <div style={{ fontSize: '1.8rem', fontWeight: 800, color: result.prediction === 1 ? '#f87171' : '#34d399' }}>
                    {result.churn_probability_pct}%
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Probability
                  </div>
                </div>

                <div style={{ marginTop: '1rem' }}>
                  <span style={{
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    padding: '0.35rem 1rem',
                    borderRadius: '20px',
                    background: result.prediction === 1 ? 'rgba(239,68,68,0.2)' : 'rgba(52,211,153,0.2)',
                    color: result.prediction === 1 ? '#f87171' : '#34d399',
                    border: `1px solid ${result.prediction === 1 ? '#ef4444' : '#34d399'}`
                  }}>
                    {result.risk_level.toUpperCase()}
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '1.2rem', textAlign: 'center' }}>
                  <div style={{ background: '#f8fafc', padding: '0.6rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>MONTHLY SPEND</div>
                    <div style={{ fontSize: '1rem', fontWeight: 700, color: '#0f172a', marginTop: '0.2rem' }}>
                      {formatINR(result.monthly_spend)}
                    </div>
                  </div>
                  <div style={{ background: '#f8fafc', padding: '0.6rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>ANNUAL AT RISK</div>
                    <div style={{ fontSize: '1rem', fontWeight: 700, color: result.prediction === 1 ? '#dc2626' : '#059669', marginTop: '0.2rem' }}>
                      {formatINR(result.annual_revenue_at_risk)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Recommendations */}
              <div style={{ marginTop: 'auto', background: '#f8fafc', padding: '1.2rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.5rem' }}>
                  💡 Prescriptive Retention Strategy:
                </div>
                <div style={{ fontSize: '0.82rem', color: '#475569', lineHeight: '1.5' }}>
                  {result.recommendation}
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-card" style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '3rem' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.6 }}>🔮</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.5rem' }}>
                Awaiting Customer Data
              </div>
              <p style={{ fontSize: '0.82rem', color: '#64748b', maxWidth: '320px', lineHeight: 1.5 }}>
                Configure the customer features on the left or select a preset and click "Generate Churn Prediction" to run the exact ML pipeline.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
