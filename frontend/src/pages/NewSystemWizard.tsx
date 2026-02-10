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
  Lightbulb,
  Sun,
  Wind,
  Battery,
  Thermometer,
  Droplets,
  Radio,
  Ship,
  Train,
  Leaf,
  Fuel,
  Wifi,
  Shield,
  Component,
  Eye,
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

interface EnrichedField {
  name: string;
  display_name?: string;
  description?: string;
  type?: string;
  physical_unit?: string | null;
  physical_unit_full?: string | null;
  category?: string;
  component?: string | null;
  engineering_context?: {
    typical_range?: { min: number; max: number } | null;
    operating_range_description?: string | null;
    what_high_means?: string | null;
    what_low_means?: string | null;
    safety_critical?: boolean;
    design_limit_hint?: { min: number; max: number } | null;
  };
  value_interpretation?: {
    assessment?: string;
  };
  confidence?: { type: number; unit: number; meaning: number; overall: number } | number;
  reasoning?: string;
  source_file?: string;
  sample_values?: unknown[];
  // Legacy compat
  inferred_type?: string;
  inferred_meaning?: string;
  field_category?: string;
}

interface FieldRelationshipV2 {
  fields: string[];
  relationship: string;
  description: string;
  expected_correlation?: string;
  diagnostic_value?: string;
}

interface ConfirmationRequest {
  field?: string;
  field_name?: string;
  reason?: string;
  question: string;
  type?: string;
  inferred_unit?: string | null;
  inferred_type?: string;
  sample_values?: unknown[];
}

interface AIRecommendation {
  suggested_name: string;
  suggested_type: string;
  suggested_description: string;
  confidence: number;
  reasoning?: string;
  system_subtype?: string | null;
  domain?: string | null;
  detected_components?: { name: string; role: string; fields: string[] }[];
  probable_use_case?: string | null;
  data_characteristics?: {
    temporal_resolution?: string;
    duration_estimate?: string;
    completeness?: string;
  };
  analysis_summary?: {
    files_analyzed: number;
    total_records: number;
    unique_fields: number;
    ai_powered?: boolean;
  };
}

// Icon map for system types
const systemTypeIcons: Record<string, typeof Car> = {
  vehicle: Car,
  aerospace: Rocket,
  robot: Cpu,
  medical_device: Heart,
  industrial: Factory,
  energy: Zap,
  solar_energy: Sun,
  wind_energy: Wind,
  battery_system: Battery,
  hvac: Thermometer,
  water_treatment: Droplets,
  telecom: Radio,
  marine: Ship,
  rail: Train,
  agriculture: Leaf,
  semiconductor: Cpu,
  oil_gas: Fuel,
  generic_iot: Wifi,
};

// Static fallback system types (used before API response)
const defaultSystemTypes: Record<string, string> = {
  vehicle: 'Vehicle & Automotive',
  aerospace: 'Aerospace & Aviation',
  robot: 'Robotics & Automation',
  medical_device: 'Medical & Biomedical',
  industrial: 'Industrial Manufacturing',
  energy: 'Energy & Power Systems',
  solar_energy: 'Solar & Photovoltaic',
  wind_energy: 'Wind Energy',
  battery_system: 'Battery & Energy Storage',
  hvac: 'HVAC & Building Systems',
  water_treatment: 'Water & Fluid Systems',
  telecom: 'Telecommunications',
  marine: 'Marine & Maritime',
  rail: 'Rail & Railway',
  agriculture: 'Agriculture & AgTech',
  semiconductor: 'Semiconductor & Cleanroom',
  oil_gas: 'Oil, Gas & Pipeline',
  generic_iot: 'Generic IoT / Sensor Network',
};

// Steps - new order
const steps = [
  { id: 1, name: 'Upload Data', description: 'Add files to the data pool' },
  { id: 2, name: 'System Settings', description: 'Smart-detected configuration' },
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
  const [availableSystemTypes, setAvailableSystemTypes] = useState<Record<string, string>>(defaultSystemTypes);

  // Step 3: Discovered schema
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [createdSystemId, setCreatedSystemId] = useState<string | null>(null);
  const [discoveredFields, setDiscoveredFields] = useState<EnrichedField[]>([]);
  const [fieldRelationships, setFieldRelationships] = useState<FieldRelationshipV2[]>([]);
  const [blindSpots, setBlindSpots] = useState<string[]>([]);
  const [confirmationRequests, setConfirmationRequests] = useState<ConfirmationRequest[]>([]);
  const [recordCount, setRecordCount] = useState(0);
  const [aiPowered, setAiPowered] = useState(false);

  // Step 4: Confirmations
  const [confirmations, setConfirmations] = useState<Record<string, { confirmed: boolean; correctedValue?: string }>>({});

  // Get confirmation requests (new format uses field, legacy uses field_name)
  const fieldConfirmations = confirmationRequests.filter(
    (req) => req.field || req.field_name || req.type === 'field_confirmation'
  );

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
        return fieldConfirmations.length === 0 || Object.keys(confirmations).length === fieldConfirmations.length;
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

        // Store analysis ID for later use
        setAnalysisId(result.analysis_id);

        // Set AI recommendations
        setAiRecommendation(result.recommendation);
        setAiPowered(result.ai_powered || false);

        // Update available system types from API
        if (result.available_system_types) {
          setAvailableSystemTypes(result.available_system_types);
        }

        // Pre-fill form with AI suggestions
        setSystemData({
          name: result.recommendation?.suggested_name || '',
          system_type: result.recommendation?.suggested_type || '',
          serial_number: '',
          model: '',
          description: result.recommendation?.suggested_description || '',
        });

        // Store discovered fields (now LLM-enriched)
        setDiscoveredFields(result.discovered_fields || []);
        setFieldRelationships(result.field_relationships || []);
        setBlindSpots(result.blind_spots || []);
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
      // Data is already analyzed and stored - just associate it with the new system
      setIsProcessing(true);
      setError(null);
      try {
        const created = await systemsApi.create({
          name: systemData.name,
          system_type: systemData.system_type,
          serial_number: systemData.serial_number || undefined,
          model: systemData.model || undefined,
          analysis_id: analysisId || undefined,  // Associate pre-analyzed data
        });
        setCreatedSystemId(created.id);

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
            <div className="bg-stone-700/50 rounded-lg p-4 border border-stone-600">
              <div className="flex items-start gap-3">
                <Database className="w-5 h-5 text-primary-400 mt-0.5" />
                <div>
                  <h3 className="font-medium text-white">Data Pool</h3>
                  <p className="text-sm text-stone-400 mt-1">
                    Upload all relevant data files. Our system will automatically discover relationships
                    between files, understand the data structure, and infer the system type.
                    You can upload telemetry data, logs, and description files that explain the fields.
                  </p>
                </div>
              </div>
            </div>

            <div
              className={clsx(
                'border-2 border-dashed rounded-xl p-8 text-center transition-all',
                'border-stone-600 hover:border-stone-500 cursor-pointer'
              )}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
            >
              <input
                type="file"
                className="hidden"
                id="file-upload"
                accept=".csv,.tsv,.txt,.dat,.log,.json,.jsonl,.ndjson,.xml,.yaml,.yml,.parquet,.feather,.xlsx,.xls,.can,.bin"
                multiple
                onChange={handleFileSelect}
              />
              <label htmlFor="file-upload" className="cursor-pointer block">
                <Upload className="w-12 h-12 text-stone-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-white">Drop files here or click to browse</p>
                <p className="text-stone-400 mt-1">
                  Upload multiple files at once
                </p>
                <p className="text-sm text-stone-400 mt-2">
                  Supported: CSV, JSON, JSONL, Parquet, Excel, Text, Markdown, Log files
                </p>
              </label>
            </div>

            {/* Uploaded files list */}
            {uploadedFiles.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-stone-300">
                  Files in Data Pool ({uploadedFiles.length})
                </h4>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {uploadedFiles.map((uf) => {
                    const FileIcon = getFileIcon(uf.file.name);
                    return (
                      <div
                        key={uf.id}
                        className="flex items-center gap-3 p-3 bg-stone-700/50 rounded-lg border border-stone-600"
                      >
                        <FileIcon className="w-5 h-5 text-primary-400" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-white truncate">{uf.file.name}</p>
                          <p className="text-xs text-stone-400">
                            {(uf.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                        <button
                          onClick={() => removeFile(uf.id)}
                          className="p-1 hover:bg-stone-500 rounded transition-colors"
                        >
                          <X className="w-4 h-4 text-stone-400 hover:text-red-400" />
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
              <div className={clsx(
                'rounded-lg p-4 border',
                aiPowered
                  ? 'bg-primary-500/10 border-primary-500/30'
                  : 'bg-stone-700/50 border-stone-600'
              )}>
                <div className="flex items-start gap-3">
                  {aiPowered ? (
                    <Brain className="w-5 h-5 text-primary-400 mt-0.5" />
                  ) : (
                    <Lightbulb className="w-5 h-5 text-stone-400 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <h3 className={clsx('font-medium', aiPowered ? 'text-primary-400' : 'text-stone-300')}>
                      {aiPowered ? 'AI-Powered Analysis Complete' : 'Rule-Based Analysis Complete'}
                    </h3>
                    <p className="text-sm text-stone-300 mt-1">
                      {aiRecommendation.suggested_description || aiRecommendation.reasoning}
                    </p>
                    {aiRecommendation.probable_use_case && (
                      <p className="text-sm text-stone-400 mt-1">
                        Use case: {aiRecommendation.probable_use_case}
                      </p>
                    )}
                    <div className="flex items-center gap-4 mt-2">
                      <span className="text-xs text-stone-400">
                        Confidence: {(aiRecommendation.confidence * 100).toFixed(0)}%
                      </span>
                      {aiRecommendation.system_subtype && (
                        <span className="text-xs text-primary-300 bg-primary-500/10 px-2 py-0.5 rounded">
                          {aiRecommendation.system_subtype}
                        </span>
                      )}
                      {aiRecommendation.data_characteristics?.temporal_resolution && (
                        <span className="text-xs text-stone-400">
                          Resolution: {aiRecommendation.data_characteristics.temporal_resolution}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-stone-300 mb-2">
                System Name *
                {aiRecommendation && (
                  <span className="ml-2 text-xs text-primary-400">(Smart Detection)</span>
                )}
              </label>
              <input
                type="text"
                value={systemData.name}
                onChange={(e) => setSystemData((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., Fleet Vehicle Alpha, Robot Arm Unit 7"
                className="w-full px-4 py-3 bg-stone-700 border border-stone-600 rounded-lg text-white placeholder-stone-400 focus:outline-none focus:border-primary-500 transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-stone-300 mb-3">
                System Type *
                {aiRecommendation && (
                  <span className="ml-2 text-xs text-primary-400">(Smart Detection)</span>
                )}
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-80 overflow-y-auto pr-1">
                {Object.entries(availableSystemTypes).map(([typeId, displayName]) => {
                  const IconComp = systemTypeIcons[typeId] || Component;
                  const isRecommended = aiRecommendation?.suggested_type === typeId;
                  return (
                    <button
                      key={typeId}
                      onClick={() => setSystemData((prev) => ({ ...prev, system_type: typeId }))}
                      className={clsx(
                        'p-4 rounded-lg border-2 text-left transition-all relative',
                        systemData.system_type === typeId
                          ? 'border-primary-500 bg-primary-500/10'
                          : isRecommended
                          ? 'border-primary-500/40 bg-primary-500/5 hover:border-primary-500'
                          : 'border-stone-600 bg-stone-700/50 hover:border-stone-500'
                      )}
                    >
                      {isRecommended && (
                        <span className="absolute -top-2 -right-2 px-1.5 py-0.5 bg-primary-500 text-white text-[10px] font-bold rounded">
                          AI
                        </span>
                      )}
                      <IconComp
                        className={clsx(
                          'w-5 h-5 mb-2',
                          systemData.system_type === typeId ? 'text-primary-400' : 'text-stone-400'
                        )}
                      />
                      <p className="font-medium text-white text-sm">{displayName}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-stone-300 mb-2">
                Description
                {aiRecommendation && (
                  <span className="ml-2 text-xs text-primary-400">(Smart Detection)</span>
                )}
              </label>
              <textarea
                value={systemData.description}
                onChange={(e) => setSystemData((prev) => ({ ...prev, description: e.target.value }))}
                placeholder="Brief description of the system..."
                rows={4}
                className="w-full px-4 py-3 bg-stone-700 border border-stone-600 rounded-lg text-white placeholder-stone-400 focus:outline-none focus:border-primary-500 transition-colors resize-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-stone-300 mb-2">
                  Serial Number (Optional)
                </label>
                <input
                  type="text"
                  value={systemData.serial_number}
                  onChange={(e) => setSystemData((prev) => ({ ...prev, serial_number: e.target.value }))}
                  placeholder="e.g., VH-2024-001"
                  className="w-full px-4 py-3 bg-stone-700 border border-stone-600 rounded-lg text-white placeholder-stone-400 focus:outline-none focus:border-primary-500 transition-colors"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-stone-300 mb-2">
                  Model (Optional)
                </label>
                <input
                  type="text"
                  value={systemData.model}
                  onChange={(e) => setSystemData((prev) => ({ ...prev, model: e.target.value }))}
                  placeholder="e.g., EV-X1"
                  className="w-full px-4 py-3 bg-stone-700 border border-stone-600 rounded-lg text-white placeholder-stone-400 focus:outline-none focus:border-primary-500 transition-colors"
                />
              </div>
            </div>
          </div>
        );

      case 3: {
        const categoryInfo: Record<string, { label: string; color: string; bgColor: string; description: string }> = {
          content: { label: 'Content', color: 'text-green-400', bgColor: 'bg-green-500/15 border-green-500/30', description: 'Core measurement/sensor data for analysis' },
          temporal: { label: 'Temporal', color: 'text-blue-400', bgColor: 'bg-blue-500/15 border-blue-500/30', description: 'Time-related fields for trend tracking' },
          identifier: { label: 'Identifier', color: 'text-yellow-400', bgColor: 'bg-yellow-500/15 border-yellow-500/30', description: 'IDs and labels for grouping' },
          auxiliary: { label: 'Auxiliary', color: 'text-stone-400', bgColor: 'bg-stone-600/30 border-stone-500/30', description: 'Low-value or redundant fields' },
        };

        // Use new 'category' field, fallback to legacy 'field_category'
        const getCategory = (f: EnrichedField) => f.category || f.field_category || 'content';
        const contentFields = discoveredFields.filter((f) => getCategory(f) === 'content');
        const temporalFields = discoveredFields.filter((f) => getCategory(f) === 'temporal');
        const identifierFields = discoveredFields.filter((f) => getCategory(f) === 'identifier');
        const auxiliaryFields = discoveredFields.filter((f) => getCategory(f) === 'auxiliary');

        // Get overall confidence for a field (handle both object and number formats)
        const getOverallConf = (f: EnrichedField): number => {
          if (typeof f.confidence === 'number') return f.confidence;
          if (f.confidence && typeof f.confidence === 'object') return f.confidence.overall || 0.5;
          return 0.5;
        };

        const renderFieldCard = (field: EnrichedField, info: typeof categoryInfo[string]) => {
          const conf = getOverallConf(field);
          const displayName = field.display_name || field.name;
          const description = field.description || field.inferred_meaning || 'Unknown';
          const fieldType = field.type || field.inferred_type || 'unknown';
          const unit = field.physical_unit;
          const ec = field.engineering_context;

          return (
            <div
              key={field.name}
              className={clsx('rounded-lg p-4 border', info.bgColor)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1 min-w-0">
                  {ec?.safety_critical ? (
                    <Shield className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
                  ) : (
                    <Database className="w-4 h-4 text-stone-400 mt-0.5 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="font-medium text-white text-sm">{displayName}</p>
                    <p className="font-mono text-xs text-stone-400">{field.name}</p>
                    <p className="text-sm text-stone-300 mt-1">{description}</p>
                    {field.component && (
                      <span className="inline-block mt-1 px-2 py-0.5 bg-stone-600/50 rounded text-xs text-stone-300">
                        {field.component}
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-right shrink-0 ml-3">
                  <span className={clsx('text-sm font-medium', getConfidenceColor(conf))}>
                    {(conf * 100).toFixed(0)}%
                  </span>
                  <div className="flex items-center gap-2 mt-1 justify-end">
                    <span className="px-2 py-0.5 bg-stone-700 rounded text-xs text-stone-300">
                      {fieldType}
                    </span>
                    {unit && (
                      <span className="px-2 py-0.5 bg-primary-500/20 rounded text-xs text-primary-300">
                        {unit}
                      </span>
                    )}
                    {ec?.safety_critical && (
                      <span className="px-2 py-0.5 bg-red-500/20 rounded text-xs text-red-300">
                        Safety
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Engineering context (collapsed by default, shown for AI-powered) */}
              {aiPowered && ec && (ec.what_high_means || ec.operating_range_description) && (
                <div className="mt-3 pt-3 border-t border-stone-600/50 space-y-1">
                  {ec.operating_range_description && (
                    <p className="text-xs text-stone-400">{ec.operating_range_description}</p>
                  )}
                  {ec.what_high_means && (
                    <p className="text-xs text-stone-400">
                      <span className="text-stone-500">High:</span> {ec.what_high_means}
                    </p>
                  )}
                  {ec.what_low_means && (
                    <p className="text-xs text-stone-400">
                      <span className="text-stone-500">Low:</span> {ec.what_low_means}
                    </p>
                  )}
                </div>
              )}

              {/* Value assessment */}
              {field.value_interpretation?.assessment && (
                <div className="mt-2 pt-2 border-t border-stone-600/50">
                  <p className="text-xs text-stone-400">{field.value_interpretation.assessment}</p>
                </div>
              )}

              {/* Sample values fallback */}
              {!field.value_interpretation?.assessment && field.sample_values && field.sample_values.length > 0 && (
                <div className="mt-2 pt-2 border-t border-stone-600/50">
                  <p className="text-xs text-stone-400">
                    Samples: <span className="font-mono text-stone-300">{field.sample_values.slice(0, 4).join(', ')}</span>
                  </p>
                </div>
              )}
            </div>
          );
        };

        const renderFieldGroup = (fields: EnrichedField[], category: string) => {
          if (fields.length === 0) return null;
          const info = categoryInfo[category];
          return (
            <div key={category}>
              <div className="flex items-center gap-2 mb-3">
                <span className={clsx('text-sm font-semibold', info.color)}>{info.label}</span>
                <span className="text-xs text-stone-400">({fields.length} fields)</span>
                <span className="text-xs text-stone-400">— {info.description}</span>
              </div>
              <div className="space-y-2">
                {fields.map((field) => renderFieldCard(field, info))}
              </div>
            </div>
          );
        };

        return (
          <div className="space-y-6">
            {/* Header */}
            <div className={clsx(
              'rounded-lg p-4 border',
              aiPowered ? 'bg-primary-500/10 border-primary-500/30' : 'bg-green-500/10 border-green-500/30'
            )}>
              <div className="flex items-center gap-3">
                {aiPowered ? <Brain className="w-5 h-5 text-primary-400" /> : <Sparkles className="w-5 h-5 text-green-400" />}
                <div>
                  <h3 className={clsx('font-medium', aiPowered ? 'text-primary-400' : 'text-green-400')}>
                    {aiPowered ? 'AI-Powered Schema Discovery' : 'Schema Discovery Complete'}
                  </h3>
                  <p className="text-sm text-stone-300">
                    Analyzed {recordCount.toLocaleString()} records from {uploadedFiles.length} files
                    and discovered {discoveredFields.length} fields
                  </p>
                </div>
              </div>
            </div>

            {/* Detected components (AI only) */}
            {aiPowered && aiRecommendation?.detected_components && aiRecommendation.detected_components.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-stone-300 mb-2">Detected Components</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {aiRecommendation.detected_components.map((comp, idx) => (
                    <div key={idx} className="bg-stone-700/50 rounded-lg p-3 border border-stone-600">
                      <p className="font-medium text-white text-sm">{comp.name}</p>
                      <p className="text-xs text-stone-400">{comp.role}</p>
                      <p className="text-xs text-stone-500 mt-1">{comp.fields.length} fields</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Category summary */}
            <div className="grid grid-cols-4 gap-3">
              {[
                { cat: 'content', count: contentFields.length },
                { cat: 'temporal', count: temporalFields.length },
                { cat: 'identifier', count: identifierFields.length },
                { cat: 'auxiliary', count: auxiliaryFields.length },
              ].map(({ cat, count }) => {
                const info = categoryInfo[cat];
                return (
                  <div key={cat} className="bg-stone-700/50 rounded-lg p-3 border border-stone-600 text-center">
                    <p className={clsx('text-lg font-bold', info.color)}>{count}</p>
                    <p className="text-xs text-stone-400">{info.label}</p>
                  </div>
                );
              })}
            </div>

            {/* Field groups */}
            <div className="space-y-6">
              {renderFieldGroup(contentFields, 'content')}
              {renderFieldGroup(temporalFields, 'temporal')}
              {renderFieldGroup(identifierFields, 'identifier')}
              {renderFieldGroup(auxiliaryFields, 'auxiliary')}
            </div>

            {/* Relationships (AI only) */}
            {aiPowered && fieldRelationships.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-stone-300 mb-2">Field Relationships</h4>
                <div className="space-y-2">
                  {fieldRelationships.slice(0, 5).map((rel, idx) => (
                    <div key={idx} className="bg-stone-700/30 rounded-lg p-3 border border-stone-600/50">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-primary-300">{rel.fields.join(' ↔ ')}</span>
                        <span className="px-2 py-0.5 bg-stone-600 rounded text-xs text-stone-300">{rel.relationship}</span>
                      </div>
                      <p className="text-xs text-stone-400 mt-1">{rel.description}</p>
                      {rel.diagnostic_value && (
                        <p className="text-xs text-stone-500 mt-1">{rel.diagnostic_value}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Blind spots (AI only) */}
            {aiPowered && blindSpots.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-stone-300 mb-2 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-yellow-400" />
                  Blind Spots Detected
                </h4>
                <div className="space-y-1">
                  {blindSpots.map((spot, idx) => (
                    <div key={idx} className="bg-yellow-500/5 border border-yellow-500/20 rounded-lg p-3">
                      <p className="text-xs text-stone-300">{spot}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      }

      case 4:
        return (
          <div className="space-y-6">
            <div className="bg-stone-700/50 rounded-lg p-4 border border-stone-600">
              <div className="flex items-start gap-3">
                <HelpCircle className="w-5 h-5 text-primary-400 mt-0.5" />
                <div>
                  <h3 className="font-medium text-white">Human-in-the-Loop Confirmation</h3>
                  <p className="text-sm text-stone-400 mt-1">
                    Please verify the system's understanding of your data. This ensures accuracy
                    and builds trust in the analysis.
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {fieldConfirmations.map((req, idx) => {
                const fieldName = req.field || req.field_name || `field_${idx}`;
                return (
                  <div
                    key={fieldName}
                    className={clsx(
                      'p-5 rounded-lg border-2 transition-all',
                      confirmations[fieldName]?.confirmed === true
                        ? 'border-green-500/50 bg-green-500/5'
                        : confirmations[fieldName]?.confirmed === false
                        ? 'border-orange-500/50 bg-orange-500/5'
                        : 'border-stone-600 bg-stone-700/50'
                    )}
                  >
                    <p className="text-white mb-2">{req.question}</p>
                    {req.reason && (
                      <p className="text-sm text-stone-400 mb-4">{req.reason}</p>
                    )}

                    {/* Show type/unit/samples if available (legacy or enriched) */}
                    {(req.inferred_type || req.inferred_unit || req.sample_values) && (
                      <div className="bg-stone-700 rounded-lg p-3 mb-4">
                        <div className="flex items-center gap-4 text-sm">
                          {req.inferred_type && (
                            <div>
                              <span className="text-stone-400">Type: </span>
                              <span className="text-stone-300">{req.inferred_type}</span>
                            </div>
                          )}
                          {req.inferred_unit && (
                            <div>
                              <span className="text-stone-400">Unit: </span>
                              <span className="text-primary-300">{req.inferred_unit}</span>
                            </div>
                          )}
                        </div>
                        {req.sample_values && req.sample_values.length > 0 && (
                          <div className="mt-2">
                            <span className="text-stone-400 text-sm">Samples: </span>
                            <span className="text-stone-300 font-mono text-sm">
                              {req.sample_values.slice(0, 3).join(', ')}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    <div className="flex gap-3">
                      <button
                        onClick={() => handleConfirmation(fieldName, true)}
                        className={clsx(
                          'flex-1 px-4 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2',
                          confirmations[fieldName]?.confirmed === true
                            ? 'bg-green-500 text-white'
                            : 'bg-stone-700 text-white hover:bg-stone-500'
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
                            : 'bg-stone-700 text-white hover:bg-stone-500'
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

            {fieldConfirmations.length > 0 && (
              <div className="text-center text-sm text-stone-400">
                {Object.keys(confirmations).length} of {fieldConfirmations.length} fields confirmed
              </div>
            )}

            {fieldConfirmations.length === 0 && (
              <div className="text-center py-8">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
                <p className="text-stone-300 font-medium">No confirmations needed</p>
                <p className="text-stone-400 text-sm">System is confident about all discovered fields</p>
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
            <p className="text-stone-400 mb-8 max-w-md mx-auto">
              <span className="text-white font-medium">{systemData.name}</span> has been created and
              configured. The system is now ready to analyze your data.
            </p>

            <div className="grid grid-cols-3 gap-4 max-w-lg mx-auto mb-8">
              <div className="bg-stone-700 rounded-lg p-4 border border-stone-600">
                <Database className="w-6 h-6 text-primary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{recordCount.toLocaleString()}</p>
                <p className="text-xs text-stone-400">Records</p>
              </div>
              <div className="bg-stone-700 rounded-lg p-4 border border-stone-600">
                <FileText className="w-6 h-6 text-primary-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{discoveredFields.length}</p>
                <p className="text-xs text-stone-400">Fields</p>
              </div>
              <div className="bg-stone-700 rounded-lg p-4 border border-stone-600">
                <Check className="w-6 h-6 text-green-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{uploadedFiles.length}</p>
                <p className="text-xs text-stone-400">Files</p>
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
                className="px-6 py-3 bg-stone-700 hover:bg-stone-500 text-white rounded-lg font-medium transition-colors"
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
          className="flex items-center gap-2 text-stone-400 hover:text-white mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Systems
        </button>
        <h1 className="text-3xl font-bold text-white">Add New System</h1>
        <p className="text-stone-400 mt-1">Upload your data and let smart detection configure your system</p>
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
                      : 'bg-stone-700 text-stone-400'
                  )}
                >
                  {currentStep > step.id ? <Check className="w-5 h-5" /> : step.id}
                </div>
                <div className="mt-2 text-center">
                  <p
                    className={clsx(
                      'text-sm font-medium',
                      currentStep >= step.id ? 'text-white' : 'text-stone-400'
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
                    currentStep > step.id ? 'bg-green-500' : 'bg-stone-700'
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
        <div className="bg-stone-700 rounded-xl border border-stone-600 p-8">
          {renderStepContent()}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <div>
                <h3 className="font-medium text-red-400">Error</h3>
                <p className="text-sm text-stone-300">{error}</p>
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
                  ? 'bg-stone-700 text-stone-400 cursor-not-allowed'
                  : 'bg-stone-700 text-white hover:bg-stone-500'
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
                  : 'bg-stone-700 text-stone-400 cursor-not-allowed'
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
