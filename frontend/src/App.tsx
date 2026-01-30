import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Systems from './pages/Systems';
import NewSystemWizard from './pages/NewSystemWizard';
import SystemDetail from './pages/SystemDetail';
import DataIngestion from './pages/DataIngestion';
import Conversation from './pages/Conversation';
import Settings from './pages/Settings';
import AnomalyExplorer from './pages/AnomalyExplorer';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="systems" element={<Systems />} />
        <Route path="systems/new" element={<NewSystemWizard />} />
        <Route path="systems/:systemId" element={<SystemDetail />} />
        <Route path="systems/:systemId/ingest" element={<DataIngestion />} />
        <Route path="systems/:systemId/chat" element={<Conversation />} />
        <Route path="systems/:systemId/anomalies" element={<AnomalyExplorer />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}

export default App;
