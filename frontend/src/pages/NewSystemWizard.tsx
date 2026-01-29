import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Upload,
  FileText,
  Cpu,
  Car,
  Heart,
  Rocket,
  Factory,
  Loader2,
  CheckCircle,
  XCircle,
  HelpCircle,
  Zap,
  Database,
  Brain,
  Sparkles,
  AlertCircle,
  X,
  FileJson,
  FileSpreadsheet,
  File,
  Lightbulb
} from 'lucide-react';
import clsx from 'clsx';
import { systemsApi } from '../services/api';

// Types
interface SystemFormData {
  name: string;
  system_type: string;
  serial_number: string;
  model: string;
  description: string;
}

interface UploadedFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'uploaded' | 'error';
  error?: string;
}

interface DiscoveredField {
  name: string;
  inferred_type: string;
  physical_unit?: string | null;
  inferred_meaning?: string;
  confidence: number;
  sample_values?: unknown[];
  source_file?: string;
}

interface ConfirmationRequest {
  field_name?: string;
  question: string;
  inferred_unit?: string | null;
  inferred_type?: string;
  sample_values?: unknown[];
}

interface AIRecommendation {
  suggested_name: string;
  suggested_type: string;
  suggested_description: string;
  confidence: number;
  reasoning: string;
}

// System type options
const systemTypes = [
  { id: 'vehicle', name: 'Vehicle', icon: Car, description: 'Cars, trucks, EVs, autonomous vehicles' },
  { id: 'robot', name: 'Robot', icon: Cpu, description: 'Industrial robots, robotic arms, drones' },
  { id: 'medical_device', name: 'Medical Device', icon: Heart, description: 'MRI, CT scanners, diagnostic equipment' },
  { id: 'aerospace', name: 'Aerospace', icon: Rocket, description: 'Aircraft, satellites, propulsion systems' },
  { id: 'industrial', name: 'Industrial', icon: Factory, description: 'Manufacturing equipment, pumps, motors' },
];

// Steps - new order
const steps = [
  { id: 1, name: 'Upload Data', description: 'Add files to the data pool' },
  { id: 2, name: 'System Settings', description: 'AI-recommended configuration' },
  { id: 3, name: 'Schema Discovery', description: 'Review discovered data structure' },
  { id: 4, name: 'Confirm Fields', description: 'Verify the discovered schema' },
  { id: 5, name: 'Complete', description: 'System is ready for analysis' },
];

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (ext === 'json' || ext === 'jsonl') return FileJson;
  if (ext === 'csv' || ext === 'xlsx' || ext === 'parquet') return FileSpreadsheet;
  return File;
}

export default function NewSystemWizard() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Step 1: File uploads (multiple)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);

  // Step 2: System details (AI-recommended)
  const [systemData, setSystemData] = useState<SystemFormData>({
    name: '',
    system_type: '',
    serial_number: '',
    model: '',
    description: '',
  });
  const [aiRecommendation, setAiRecommendation] = useState<AIRecommendation | null>(null);

  // Step 3: Discovered schema
  const [createdSystemId, setCreatedSystemId] = useState<string | null>(null);
  const [discoveredFields, setDiscoveredFields] = useState<DiscoveredField[]>([]);
  const [confirmationRequests, setConfirmationRequests] = useState<ConfirmationRequest[]>([]);
  const [recordCount, setRecordCount] = useState(0);

  // Step 4: Confirmations
  const [confirmations, setConfirmations] = useState<Record<string, { confirmed: boolean; correctedValue?: string }>>({});

  // Navigation
  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return uploadedFiles.length > 0;
      case 2:
        return systemData.name.trim() !== '' && systemData.system_type !== '';
      case 3:
        return discoveredFields.length > 0;
      case 4:
        return Object.keys(confirmations).length === confirmationRequests.length;
      default:
        return true;
    }
  };

  const handleNext = async () => {
    if (currentStep === 1) {
      // Analyze uploaded files and get AI recommendations
      setIsProcessing(true);
      setError(null);
      try {
        // First, upload all files and analyze them
        const formData = new FormData();
        uploadedFiles.forEach((uf) => {
          formData.append('files', uf.file);
        });

        const response = await fetch('/api/v1/systems/analyze-files', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Failed to analyze files');
        }

        const result = await response.json();

        // Set AI recommendations
        setAiRecommendation(result.recommendation);

        // Pre-fill form with AI suggestions
        setSystemData({
          name: result.recommendation?.suggested_name || '',
          system_type: result.recommendation?.suggested_type || '',
          serial_number: '',
          model: '',
          description: result.recommendation?.suggested_description || '',
        });

        // Store discovered fields for later
        setDiscoveredFields(result.discovered_fields || []);
        setConfirmationRequests(result.confirmation_requests || []);
        setRecordCount(result.total_records || 0);

        setCurrentStep(2);
      } catch (err) {
        console.error('Failed to analyze files:', err);
        setError('Failed to analyze files. Please make sure the backend is running.');
      } finally {
        setIsProcessing(false);
      }
    } else if (currentStep === 2) {
      // Create the system with user-confirmed settings
      setIsProcessing(true);
      setError(null);
      try {
        const created = await systemsApi.create({
          name: systemData.name,
          system_type: systemData.system_type,
          serial_number: systemData.serial_number || undefined,
          model: systemData.model || undefined,
        });
        setCreatedSystemId(created.id);

        // Now ingest all the uploaded files
        for (const uf of uploadedFiles) {
          await systemsApi.ingest(created.id, uf.file, uf.file.name);
        }

        setCurrentStep(3);
      } catch (err) {
        console.error('Failed to create system:', err);
        setError('Failed to create system. Please try again.');
      } finally {
        setIsProcessing(false);
      }
    } else if (currentStep === 3) {
      setCurrentStep(4);
    } else if (currentStep === 4) {
      // Submit confirmations
      if (!createdSystemId) return;
      setIsProcessing(true);

      try {
        const confirmationData = Object.entries(confirmations).map(([fieldName, conf]) => ({
          field_name: fieldName,
          is_correct: conf.confirmed,
          confirmed_value: conf.correctedValue,
        }));
        await systemsApi.confirmFields(createdSystemId, confirmationData);
        setCurrentStep(5);
      } catch (err) {
        console.error('Failed to confirm fields:', err);
        setCurrentStep(5);
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  // File handling - multiple files
  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    addFiles(droppedFiles);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    addFiles(selectedFiles);
  };

  const addFiles = (files: File[]) => {
    const newFiles: UploadedFile[] = files.map((file) => ({
      file,
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      status: 'pending',
    }));
    setUploadedFiles((prev) => [...prev, ...newFiles]);
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  // Confirmation handling
  const handleConfirmation = (fieldName: string, confirmed: boolean) => {
    setConfirmations((prev) => ({
      ...prev,
      [fieldName]: { confirmed },
    }));
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-400';
    if (confidence >= 0.7) return 'text-yellow-400';
    return 'text-orange-400';
  };

  // Render step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-start gap-3">
                <Database className="w-5 h-5 text-primary-400 mt-0.5" />
                <div>
                  <h3 className="font-medium text-white">Data Pool</h3>
                  <p className="text-sm text-slate-400 mt-1">
                    Upload all relevant data files. Our AI will automatically discover relationships
                    between files, understand the data structure, and infer the system type.
                    You can upload telemetry data, logs, configuration files, or even documentation.
                  </p>
                </div>
              </div>
            </div>

            <div
              className={clsx(
                'border-2 border-dashed rounded-xl p-8 text-center transition-all',
                'border-slate-600 hover:border-slate-500 cursor-pointer'
              )}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
            >
              <input
                type="file"
                className="hidden"
                id="file-upload"
                accept=".csv,.json,.jsonl,.parquet,.xlsx,.txt,.md,.log"
                multiple
                onChange={handleFileSelect}
              />
              <label htmlFor="file-upload" className="cursor-pointer block">
                <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-white">Drop files here or click to browse</p>
                <p className="text-slate-400 mt-1">
                  Upload multiple files at once
                </p>
                <p className="text-sm text-slate-500 mt-2">
                  Supported: CSV, JSON, JSONL, Parquet, Excel, Text, Markdown, Log files
                </p>
              </label>
            </div>

            {/* Uploaded files list */}
            {uploadedFiles.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-slate-300">
                  Files in Data Pool ({uploadedFiles.length})
                </h4>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {uploadedFiles.map((uf) => {
                    const FileIcon = getFileIcon(uf.file.name);
                    return (
                      <div
                        key={uf.id}
                        className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-lg border border-slate-700"
                      >
                        <FileIcon className="w-5 h-5 text-primary-400" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-white truncate">{uf.file.name}</p>
                          <p className="text-xs text-slate-400">
                            {(uf.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                        <button
                          onClick={() => removeFile(uf.id)}
                          className="p-1 hover:bg-slate-700 rounded transition-colors"
                        >
                          <X className="w-4 h-4 text-slate-400 hover:text-red-400" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            {/* AI Recommendation Banner */}
            {aiRecommendation && (
              <div className="bg-primary-500/10 border border-primary-500/30 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <Lightbulb className="w-5 h-5 text-primary-400 mt-0.5" />
                  <div>
                    <h3 className="font-medium text-primary-400">AI Analysis Complete</h3>
                    <p className="text-sm text-slate-300 mt-1">
                      {aiRecommendation.reasoning}
                    </p>
                    <p className="text-xs text-slate-400 mt-2">
                      Confidence: {(aiRecommendation.confidence * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                System Name *
                {aiRecommendation && (
                  <span className="ml-2 text-xs text-primary-400">(AI suggested)</span>
                )}
              </label>
              <input
                type="text"
                value={systemData.name}
                onChange={(e) => setSystemData((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., Fleet Vehicle Alpha, Robot Arm Unit 7"
                className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-3">
                System Type *
                {aiRecommendation && (
                  <span className="ml-2 text-xs text-primary-400">(AI suggested)</span>
                )}
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {systemTypes.map((type) => (
                  <button
                    key={type.id}
                    onClick={() => setSystemData((prev) => ({ ...prev, system_type: type.id }))}
                    className={clsx(
                      'p-4 rounded-lg border-2 text-left transition-all',
                      systemData.system_type === type.id
                        ? 'border-primary-500 bg-primary-500/10'
                        : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
                    )}
                  >
                    <type.icon
                      className={clsx(
                        'w-6 h-6 mb-2',
                        systemData.system_type === type.id ? 'text-primary-400' : 'text-slate-400'
                      )}
                    />
                    <p className="font-medium text-white">{type.name}</p>
                    <p className="text-xs text-slate-400 mt-1">{type.description}</p>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Description
                {aiRecommendation && (
                  <span className="ml-2 text-xs text-primary-400">(AI generated)</span>
                )}
              </label>
              <textarea
                value={systemData.description}
                onChange={(e) => setSystemData((prev) => ({ ...prev, description: e.target.value }))}
                placeholder="Brief description of the system..."
                rows={4}
                className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors resize-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Serial Number (Optional)
                </label>
                <input
                  type="text"
                  value={systemData.serial_number}
                  onChange={(e) => setSystemData((prev) => ({ ...prev, serial_number: e.target.value }))}
                  placeholder="e.g., VH-2024-001"
                  className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Model (Optional)
                </label>
                <input
                  type="text"
                  value={systemData.model}
                  onChange={(e) => setSystemData((prev) => ({ ...prev, model: e.target.value }))}
                  placeholder="e.g., EV-X1"
                  className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
                />
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <div className="flex items-center gap-3">
                <Sparkles className="w-5 h-5 text-green-400" />
                <div>
                  <h3 className="font-medium text-green-400">Schema Discovery Complete</h3>
                  <p className="text-sm text-slate-300">
                    Analyzed {recordCount.toLocaleString()} records from {uploadedFiles.length} files
                    and discovered {discoveredFields.length} fields
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Discovered Fields</h3>
              <div className="space-y-3">
                {discoveredFields.map((field) => (
                  <div
                    key={field.name}
                    className="bg-slate-900/50 rounded-lg p-4 border border-slate-700"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-slate-400" />
                        <div>
                          <p className="font-medium text-white font-mono">{field.name}</p>
                          <p className="text-sm text-slate-400">{field.inferred_meaning || 'Unknown'}</p>
                          {field.source_file && (
                            <p className="text-xs text-slate-500 mt-1">From: {field.source_file}</p>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <span className={clsx('text-sm font-medium', getConfidenceColor(field.confidence))}>
                          {(field.confidence * 100).toFixed(0)}% confident
                        </span>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                            {field.inferred_type}
                          </span>
                          {field.physical_unit && (
                            <span className="px-2 py-0.5 bg-primary-500/20 rounded text-xs text-primary-300">
                              {field.physical_unit}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-slate-700">
                      <p className="text-xs text-slate-500 mb-1">Sample values:</p>
                      <p className="text-sm text-slate-300 font-mono">
                        {(field.sample_values || []).slice(0, 3).join(', ')}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
              <div className="flex items-start gap-3">
                <HelpCircle className="w-5 h-5 text-primary-400 mt-0.5" />
                <div>
                  <h3 className="font-medium text-white">Human-in-the-Loop Confirmation</h3>
                  <p className="text-sm text-slate-400 mt-1">
                    Please verify our AI's understanding of your data. This ensures accuracy and builds
                    trust in the analysis.
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {confirmationRequests.map((req, idx) => {
                const fieldName = req.field_name || `field_${idx}`;
                return (
                  <div
                    key={fieldName}
                    className={clsx(
                      'p-5 rounded-lg border-2 transition-all',
                      confirmations[fieldName]?.confirmed === true
                        ? 'border-green-500/50 bg-green-500/5'
                        : confirmations[fieldName]?.confirmed === false
                        ? 'border-orange-500/50 bg-orange-500/5'
                        : 'border-slate-700 bg-slate-800/50'
                    )}
                  >
                    <p className="text-white mb-4">{req.question}</p>

                    <div className="bg-slate-900 rounded-lg p-3 mb-4">
                      <div className="flex items-center gap-4 text-sm">
                        <div>
                          <span className="text-slate-500">Type: </span>
                          <span className="text-slate-300">{req.inferred_type || 'unknown'}</span>
                        </div>
                        {req.inferred_unit && (
                          <div>
                            <span className="text-slate-500">Unit: </span>
                            <span className="text-primary-300">{req.inferred_unit}</span>
                          </div>
                        )}
                      </div>
                      <div className="mt-2">
                        <span className="text-slate-500 text-sm">Samples: </span>
                        <span className="text-slate-300 font-mono text-sm">
                          {(req.sample_values || []).slice(0, 3).join(', ')}
                        </span>
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <button
                        onClick={() => handleConfirmation(fieldName, true)}
                        className={clsx(
                          'flex-1 px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2',
                          confirmations[fieldName]?.confirmed === true
                            ? 'bg-green-500 text-white'
                            : 'bg-slate-700 text-white hover:bg-slate-600'
                        )}
                      >
                        <CheckCircle className="w-5 h-5" />
                        Yes, Correct
                      </button>
                      <button
                        onClick={() => handleConfirmation(fieldName, false)}
                        className={clsx(
                          'flex-1 px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2',
                          confirmations[fieldName]?.confirmed === false
                            ? 'bg-orange-500 text-white'
                            : 'bg-slate-700 text-white hover:bg-slate-600'
                        )}
                      >
                        <XCircle className="w-5 h-5" />
                        Needs Correction
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {confirmationRequests.length > 0 && (
              <div className="text-center text-sm text-slate-400">
                {Object.keys(confirmations).length} of {confirmationRequests.length} fields confirmed
              </div>
            )}

            {confirmationRequests.length === 0 && (
              <div className="text-center py-8">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
                <p className="text-slate-300 font-medium">No confirmations needed</p>
                <p className="text-slate-500 text-sm">AI is confident about all discovered fields</p>
              </div>
            )}
          </div>
        );

      case 5:
        return (
          <div className="text-center py-8">
            <div className="w-20 h-20 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
              <CheckCircle className="w-10 h-10 text-green-500" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">System Ready!</h2>
            <p className="text-slate-400 mb-8 max-w-md mx-auto">
              <span className="text-white font-medium">{systemData.name}</span> has been created and
              configured. The AI agents are now ready to analyze your data.
            </p>

            <div className="grid grid-cols-3 gap-4 max-w-lg mx-auto mb-8">
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <Database className="w-6 h-6 text-primary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{recordCount.toLocaleString()}</p>
                <p className="text-xs text-slate-400">Records</p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <FileText className="w-6 h-6 text-primary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{discoveredFields.length}</p>
                <p className="text-xs text-slate-400">Fields</p>
              </div>
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <Check className="w-6 h-6 text-green-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{uploadedFiles.length}</p>
                <p className="text-xs text-slate-400">Files</p>
              </div>
            </div>

            <div className="flex gap-4 justify-center">
              <button
                onClick={() => navigate(`/systems/${createdSystemId}`)}
                className="px-6 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
              >
                <Zap className="w-5 h-5" />
                View System & Run Analysis
              </button>
              <button
                onClick={() => navigate('/systems')}
                className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors"
              >
                Go to Systems List
              </button>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen p-8">
      {/* Header */}
      <div className="max-w-4xl mx-auto mb-8">
        <button
          onClick={() => navigate('/systems')}
          className="flex items-center gap-2 text-slate-400 hover:text-white mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Systems
        </button>
        <h1 className="text-3xl font-bold text-white">Add New System</h1>
        <p className="text-slate-400 mt-1">Upload your data and let AI configure your system</p>
      </div>

      {/* Progress Steps */}
      <div className="max-w-4xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center">
              <div className="flex flex-col items-center">
                <div
                  className={clsx(
                    'w-10 h-10 rounded-full flex items-center justify-center font-medium transition-all',
                    currentStep > step.id
                      ? 'bg-green-500 text-white'
                      : currentStep === step.id
                      ? 'bg-primary-500 text-white'
                      : 'bg-slate-700 text-slate-400'
                  )}
                >
                  {currentStep > step.id ? <Check className="w-5 h-5" /> : step.id}
                </div>
                <div className="mt-2 text-center">
                  <p
                    className={clsx(
                      'text-sm font-medium',
                      currentStep >= step.id ? 'text-white' : 'text-slate-500'
                    )}
                  >
                    {step.name}
                  </p>
                </div>
              </div>
              {idx < steps.length - 1 && (
                <div
                  className={clsx(
                    'w-full h-0.5 mx-4',
                    currentStep > step.id ? 'bg-green-500' : 'bg-slate-700'
                  )}
                  style={{ width: '80px' }}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="max-w-4xl mx-auto">
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-8">
          {renderStepContent()}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <div>
                <h3 className="font-medium text-red-400">Error</h3>
                <p className="text-sm text-slate-300">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-400 hover:text-red-300"
              >
                <XCircle className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* Navigation Buttons */}
        {currentStep < 5 && (
          <div className="flex justify-between mt-6">
            <button
              onClick={handleBack}
              disabled={currentStep === 1}
              className={clsx(
                'px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2',
                currentStep === 1
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-slate-700 text-white hover:bg-slate-600'
              )}
            >
              <ArrowLeft className="w-5 h-5" />
              Back
            </button>
            <button
              onClick={handleNext}
              disabled={!canProceed() || isProcessing}
              className={clsx(
                'px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2',
                canProceed() && !isProcessing
                  ? 'bg-primary-500 hover:bg-primary-600 text-white'
                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
              )}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
                </>
              ) : currentStep === 4 ? (
                <>
                  Complete Setup
                  <Check className="w-5 h-5" />
                </>
              ) : currentStep === 1 ? (
                <>
                  Analyze Files
                  <Brain className="w-5 h-5" />
                </>
              ) : (
                <>
                  Continue
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
