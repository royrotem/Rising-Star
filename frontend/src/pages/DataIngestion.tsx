import { useState, useCallback } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Upload,
  FileText,
  CheckCircle,
  XCircle,
  HelpCircle,
  Loader2,
  AlertCircle
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';

interface DiscoveredField {
  name: string;
  inferred_type?: string;
  physical_unit?: string | null;
  inferred_meaning?: string;
  confidence: number;
  sample_values?: unknown[];
}

interface ConfirmationRequest {
  type?: string;
  field?: string;
  field_name?: string;
  question: string;
  inferred_unit?: string | null;
  inferred_type?: string;
  sample_values?: unknown[];
  options?: string[];
}

export default function DataIngestion() {
  const { systemId } = useParams();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [discoveredFields, setDiscoveredFields] = useState<DiscoveredField[]>([]);
  const [confirmationRequests, setConfirmationRequests] = useState<ConfirmationRequest[]>([]);
  const [confirmations, setConfirmations] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setError(null);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file || !systemId) return;

    setUploading(true);
    setError(null);

    try {
      const result = await systemsApi.ingest(systemId, file, file.name);

      setDiscoveredFields(result.discovered_fields || []);
      setConfirmationRequests(result.confirmation_requests || []);
      setUploadComplete(true);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed. Please try again.';
      setError(message);
    } finally {
      setUploading(false);
    }
  };

  const handleConfirmation = (fieldName: string, confirmed: boolean) => {
    setConfirmations(prev => ({
      ...prev,
      [fieldName]: confirmed,
    }));
  };

  const handleSaveConfirmations = async () => {
    if (!systemId) return;

    setSaving(true);
    setError(null);

    try {
      const fieldConfirmations = Object.entries(confirmations).map(([field_name, is_correct]) => {
        const field = discoveredFields.find(f => f.name === field_name);
        return {
          field_name,
          is_correct,
          confirmed_type: field?.inferred_type,
          confirmed_unit: field?.physical_unit || undefined,
          confirmed_meaning: field?.inferred_meaning,
        };
      });

      await systemsApi.confirmFields(systemId, fieldConfirmations);
      navigate(`/systems/${systemId}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to save confirmations.';
      setError(message);
    } finally {
      setSaving(false);
    }
  };

  const getFieldKey = (req: ConfirmationRequest) => req.field_name || req.field || '';

  const allConfirmed = confirmationRequests.length > 0 && confirmationRequests.every(
    req => confirmations[getFieldKey(req)] !== undefined
  );

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link
          to={'/systems/' + systemId}
          className="p-2 hover:bg-stone-700 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-stone-400" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">Data Ingestion</h1>
          <p className="text-stone-400">Upload data for autonomous schema discovery</p>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* Upload Area */}
        <div className="bg-stone-700 rounded-xl border border-stone-600 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Upload Data File</h2>

          {!uploadComplete ? (
            <>
              <div
                className={clsx(
                  'border-2 border-dashed rounded-xl p-12 text-center transition-colors',
                  file ? 'border-primary-500 bg-primary-500/5' : 'border-stone-600 hover:border-stone-500'
                )}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
              >
                <input
                  type="file"
                  className="hidden"
                  id="file-upload"
                  accept=".csv,.tsv,.txt,.dat,.log,.json,.jsonl,.ndjson,.xml,.yaml,.yml,.parquet,.feather,.xlsx,.xls,.can,.bin"
                  onChange={handleFileSelect}
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 text-stone-400 mx-auto mb-4" />
                  {file ? (
                    <div>
                      <p className="text-white font-medium">{file.name}</p>
                      <p className="text-stone-400 text-sm mt-1">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-white font-medium">Drop your file here</p>
                      <p className="text-stone-400 text-sm mt-1">
                        or click to browse (CSV, JSON, Parquet, Excel, CAN, binary)
                      </p>
                    </div>
                  )}
                </label>
              </div>

              <button
                onClick={handleUpload}
                disabled={!file || uploading}
                className="w-full mt-4 px-4 py-3 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    Start Ingestion
                  </>
                )}
              </button>

              <div className="mt-6 p-4 bg-stone-700/50 rounded-lg">
                <h3 className="text-sm font-medium text-primary-400 mb-2">Zero-Knowledge Ingestion</h3>
                <p className="text-sm text-stone-400">
                  Upload your raw data files. Our AI agents will autonomously analyze the
                  structure, infer field types and relationships, and present their
                  understanding for your confirmation.
                </p>
              </div>
            </>
          ) : (
            <div className="text-center py-8">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Upload Complete</h3>
              <p className="text-stone-400 mb-6">
                Discovered {discoveredFields.length} fields. Please review and confirm.
              </p>

              <div className="text-left space-y-3">
                {discoveredFields.map((field) => (
                  <div key={field.name} className="p-3 bg-stone-700/50 rounded-lg flex items-center gap-3">
                    <FileText className="w-5 h-5 text-stone-400" />
                    <div className="flex-1">
                      <p className="text-white font-medium">{field.name}</p>
                      <p className="text-sm text-stone-400">
                        {field.inferred_meaning || 'Unknown'} ({field.physical_unit || field.inferred_type || 'unknown'})
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={clsx(
                        'text-sm',
                        field.confidence > 0.9 ? 'text-green-400' :
                        field.confidence > 0.7 ? 'text-yellow-400' : 'text-orange-400'
                      )}>
                        {(field.confidence * 100).toFixed(0)}% confident
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Confirmation Panel */}
        <div className="bg-stone-700 rounded-xl border border-stone-600 p-6">
          <div className="flex items-center gap-2 mb-4">
            <HelpCircle className="w-5 h-5 text-primary-500" />
            <h2 className="text-lg font-semibold text-white">Human-in-the-Loop Confirmation</h2>
          </div>

          {!uploadComplete ? (
            <div className="text-center py-12 text-stone-400">
              <p>Upload a file to begin schema discovery</p>
            </div>
          ) : confirmationRequests.length === 0 ? (
            <div className="text-center py-12">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
              <p className="text-white mb-4">All fields auto-confirmed with high confidence!</p>
              <Link
                to={`/systems/${systemId}`}
                className="inline-flex px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors"
              >
                Back to System
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {confirmationRequests.map((req) => (
                <div
                  key={getFieldKey(req)}
                  className={clsx(
                    'p-4 rounded-lg border transition-colors',
                    confirmations[getFieldKey(req)] === true
                      ? 'border-green-500/50 bg-green-500/5'
                      : confirmations[getFieldKey(req)] === false
                      ? 'border-red-500/50 bg-red-500/5'
                      : 'border-stone-600 bg-stone-700/50'
                  )}
                >
                  <p className="text-white mb-3">{req.question}</p>

                  <div className="mb-3 p-2 bg-stone-700 rounded text-sm">
                    <span className="text-stone-400">Sample values: </span>
                    <span className="text-white">
                      {(req.sample_values || []).slice(0, 3).join(', ')}
                    </span>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => handleConfirmation(getFieldKey(req), true)}
                      className={clsx(
                        'flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
                        confirmations[getFieldKey(req)] === true
                          ? 'bg-green-500 text-white'
                          : 'bg-stone-700 text-white hover:bg-stone-500'
                      )}
                    >
                      <CheckCircle className="w-4 h-4" />
                      Confirm
                    </button>
                    <button
                      onClick={() => handleConfirmation(getFieldKey(req), false)}
                      className={clsx(
                        'flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
                        confirmations[getFieldKey(req)] === false
                          ? 'bg-red-500 text-white'
                          : 'bg-stone-700 text-white hover:bg-stone-500'
                      )}
                    >
                      <XCircle className="w-4 h-4" />
                      Correct
                    </button>
                  </div>
                </div>
              ))}

              {allConfirmed && (
                <button
                  onClick={handleSaveConfirmations}
                  disabled={saving}
                  className="w-full px-4 py-3 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {saving ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    'Save Confirmations & Continue'
                  )}
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
