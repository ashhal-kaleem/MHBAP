export interface ClassMetrics {
  label: string
  precision: number
  recall: number
  f1: number
  support: number
}

export interface EvaluationReport {
  name: string
  accuracy: number
  macro_precision: number
  macro_recall: number
  macro_f1: number
  weighted_f1: number
  cohen_kappa: number
  mae: number
  rmse: number
  per_class: ClassMetrics[]
  confusion_matrix: number[][]
  n_samples: number
}

export interface BenchmarkResponse {
  reports: EvaluationReport[]
  n_samples: number
  seed: number
}

export interface AblationResult {
  active_modalities: string[]
  dropped_modalities: string[]
  accuracy: number
  macro_f1: number
  weighted_f1: number
  cohen_kappa: number
}

export interface AblationStudyResponse {
  n_samples: number
  seed: number
  results: AblationResult[]
  baseline_f1: number
}
