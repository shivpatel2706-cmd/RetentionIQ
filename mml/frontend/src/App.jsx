import { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import ReportPage from './pages/ReportPage';
import PredictPage from './pages/PredictPage';
import BatchPage from './pages/BatchPage';
import SimulatorPage from './pages/SimulatorPage';

export default function App() {
  const [activePage, setActivePage] = useState('home');
  const [evalData, setEvalData] = useState(null);
  const [metaData, setMetaData] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [evalRes, metaRes, healthRes] = await Promise.all([
          axios.get('/api/evaluation'),
          axios.get('/api/metadata'),
          axios.get('/api/health')
        ]);
        setEvalData(evalRes.data);
        setMetaData(metaRes.data);
        setBackendOnline(healthRes.data?.status === 'healthy');
      } catch (err) {
        console.warn('Backend API connection check:', err.message);
        setBackendOnline(false);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="app-layout">
      {/* Sidebar with Navigation */}
      <Sidebar
        activePage={activePage}
        onPageChange={setActivePage}
        modelInfo={metaData}
      />

      {/* Main Content Area */}
      <main className="main-content">
        {!backendOnline && !loading && (
          <div style={{
            background: 'rgba(234, 179, 8, 0.15)',
            border: '1px solid rgba(234, 179, 8, 0.4)',
            color: '#fbbf24',
            padding: '0.6rem 1rem',
            borderRadius: '8px',
            marginBottom: '1rem',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <span>
              ⚠️ Flask Backend on port 5000 is initializing or offline. Predictions require the backend server to be running.
            </span>
            <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>Port: 5000</span>
          </div>
        )}

        {activePage === 'home' && <HomePage evalData={evalData} metaData={metaData} />}
        {activePage === 'report' && <ReportPage evalData={evalData} metaData={metaData} />}
        {activePage === 'predict' && <PredictPage />}
        {activePage === 'batch' && <BatchPage />}
        {activePage === 'simulator' && <SimulatorPage />}
      </main>
    </div>
  );
}
