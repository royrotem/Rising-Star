import { useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft,
  Upload,
  FileText,
  CheckCircle,
  XCircle,
  HelpCircle,
  Loader2
} from 'lucide-react';
import clsx from 'clsx';

interface DiscoveredField {
  name: string;
  inferred_type: string;
  physical_unit: string | null;
  inferred_meaning: string;
  confidence: number;
  sample_values: unknown[];
}

interface ConfirmationRequest {
  type: string;
  field_name: string;
  question: string;
  inferred_unit: string | null;
  inferred_type: string;
  sample_values: unknown[];
  options: string[];
}

export default function DataIngestion() {
  const { systemId } = useParams();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [discoveredFields, setDiscoveredFields] = useState<DiscoveredField[]>([]);
  const [confirmationRequests, setConfirmationRequests] = useState<ConfirmationRequest[]>([]);
  const [confirmations, setConfirmations] = useState<Record<string, boolean>>({});

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    
    // Simulate upload and schema discovery
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock discovered fields
    setDiscoveredFields([
      {
        name: 'motor_current',
        inferred_type: 'numeric',
        physical_unit: 'amperes',
        inferred_meaning: 'Motor current measurement',
        confidence: 0.92,
        sample_values: [12.5, 13.2, 11.8, 14.1, 12.9],
      },
      {
        name: 'battery_temp',
        inferred_type: 'numeric',
        physical_unit: 'celsius',
        inferred_meaning: 'Battery temperature',
        confidence: 0.88,
        sample_values: [35.2, 36.1, 34.8, 37.2, 35.9],
      },
      {
        name: 'timestamp',
        inferred_type: 'timestamp',
        physical_unit: null,
        inferred_meaning: 'Event timestamp',
        confidence: 0.95,
        sample_values: ['2024-01-10T10:00:00Z', '2024-01-10T10:01:00Z'],
      },
    ]);
    
    setConfirmationRequests([
      {
        type: 'field_confirmation',
        field_name: 'motor_current',
        question: 'I believe "motor_current" measures electric current in Amperes. Is this correct?',
        inferred_unit: 'amperes',
        inferred_type: 'numeric',
        sample_values: [12.5, 13.2, 11.8],
        options: ['Confirm', 'Correct', 'Skip'],
      },
      {
        type: 'field_confirmation',
        field_name: 'battery_temp',
        question: 'I believe "battery_temp" is a temperature reading in Celsius. Is this correct?',
        inferred_unit: 'celsius',
        inferred_type: 'numeric',
        sample_values: [35.2, 36.1, 34.8],
        options: ['Confirm', 'Correct', 'Skip'],
      },
    ]);
    
    setUploading(false);
    setUploadComplete(true);
  };

  const handleConfirmation = (fieldName: string, confirmed: boolean) => {
    setConfirmations(prev => ({
      ...prev,
      [fieldName]: confirmed,
    }));
  };

  const allConfirmed = confirmationRequests.every(
    req => confirmations[req.field_name] !== undefined
  );

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link
          to={'/systems/' + systemId}
          className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-slate-400" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">Data Ingestion</h1>
          <p className="text-slate-400">Upload data for autonomous schema discovery</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Upload Area */}
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Upload Data File</h2>
          
          {!uploadComplete ? (
            <>
              <div
                className={clsx(
                  'border-2 border-dashed rounded-xl p-12 text-center transition-colors',
                  file ? 'border-primary-500 bg-primary-500/5' : 'border-slate-600 hover:border-slate-500'
                )}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
              >
                <input
                  type="file"
                  className="hidden"
                  id="file-upload"
                  accept=".csv,.json,.jsonl,.parquet"
                  onChange={handleFileSelect}
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                  {file ? (
                    <div>
                      <p className="text-white font-medium">{file.name}</p>
                      <p className="text-slate-400 text-sm mt-1">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-white font-medium">Drop your file here</p>
                      <p className="text-slate-400 text-sm mt-1">
                        or click to browse (CSV, JSON, Parquet)
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

              <div className="mt-6 p-4 bg-slate-900/50 rounded-lg">
                <h3 className="text-sm font-medium text-primary-400 mb-2">Zero-Knowledge Ingestion</h3>
                <p className="text-sm text-slate-400">
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
              <p className="text-slate-400 mb-6">
                Discovered {discoveredFields.length} fields. Please review and confirm.
              </p>
              
              <div className="text-left space-y-3">
                {discoveredFields.map((field) => (
                  <div key={field.name} className="p-3 bg-slate-900/50 rounded-lg flex items-center gap-3">
                    <FileText className="w-5 h-5 text-slate-400" />
                    <div className="flex-1">
                      <p className="text-white font-medium">{field.name}</p>
                      <p className="text-sm text-slate-400">
                        {field.inferred_meaning} ({field.physical_unit || field.inferred_type})
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
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
          <div className="flex items-center gap-2 mb-4">
            <HelpCircle className="w-5 h-5 text-primary-500" />
            <h2 className="text-lg font-semibold text-white">Human-in-the-Loop Confirmation</h2>
          </div>

          {!uploadComplete ? (
            <div className="text-center py-12 text-slate-400">
              <p>Upload a file to begin schema discovery</p>
            </div>
          ) : confirmationRequests.length === 0 ? (
            <div className="text-center py-12">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
              <p className="text-white">All fields confirmed!</p>
            </div>
          ) : (
            <div className="space-y-4">
              {confirmationRequests.map((req) => (
                <div 
                  key={req.field_name}
                  className={clsx(
                    'p-4 rounded-lg border transition-colors',
                    confirmations[req.field_name] === true 
                      ? 'border-green-500/50 bg-green-500/5'
                      : confirmations[req.field_name] === false
                      ? 'border-red-500/50 bg-red-500/5'
                      : 'border-slate-600 bg-slate-900/50'
                  )}
                >
                  <p className="text-white mb-3">{req.question}</p>
                  
                  <div className="mb-3 p-2 bg-slate-800 rounded text-sm">
                    <span className="text-slate-400">Sample values: </span>
                    <span className="text-white">
                      {req.sample_values.slice(0, 3).join(', ')}
                    </span>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => handleConfirmation(req.field_name, true)}
                      className={clsx(
                        'flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
                        confirmations[req.field_name] === true
                          ? 'bg-green-500 text-white'
                          : 'bg-slate-700 text-white hover:bg-slate-600'
                      )}
                    >
                      <CheckCircle className="w-4 h-4" />
                      Confirm
                    </button>
                    <button
                      onClick={() => handleConfirmation(req.field_name, false)}
                      className={clsx(
                        'flex-1 px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-2',
                        confirmations[req.field_name] === false
                          ? 'bg-red-500 text-white'
                          : 'bg-slate-700 text-white hover:bg-slate-600'
                      )}
                    >
                      <XCircle className="w-4 h-4" />
                      Correct
                    </button>
                  </div>
                </div>
              ))}

              {allConfirmed && (
                <button className="w-full px-4 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors">
                  Save Confirmations & Continue
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
