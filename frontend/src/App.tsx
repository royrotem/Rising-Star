import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import SystemDetail from './pages/SystemDetail';
import DataIngestion from './pages/DataIngestion';
import Conversation from './pages/Conversation';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="systems/:systemId" element={<SystemDetail />} />
        <Route path="systems/:systemId/ingest" element={<DataIngestion />} />
        <Route path="systems/:systemId/chat" element={<Conversation />} />
      </Route>
    </Routes>
  );
}

export default App;
