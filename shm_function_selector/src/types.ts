// Shared type definitions for the SHM Function Selector extension.
//
// Extracted from index.ts as part of the modular refactor (issue #30).

export interface SHMFunction {
  name: string;
  displayName: string;
  category: string;
  module: string;
  signature: string;
  description: string;
  docstring: string;
  parameters: Array<{
    name: string;
    type: string;
    optional: boolean;
    default: string | null;
    description?: string;
    widget?: {
      widget?: string;
      min?: number;
      max?: number;
      default?: string;
      options?: string[];
      formats?: string[];
    };
    validation?: Array<{
      type: string;
      min?: number;
      max?: number;
      options?: string[];
      formats?: string[];
    }>;
  }>;
  guiMetadata?: {
    category?: string;
    complexity?: string;
    data_type?: string;
    output_type?: string;
    matlab_equivalent?: string;
    verbose_call?: string;
  };
  returns?: Array<{
    name: string;
    type: string;
    description: string;
  }>;
}

export interface CategoryNode {
  name: string;
  children: Map<string, CategoryNode>;
  functions: SHMFunction[];
  level: number;
}

export interface ParameterContext {
  parameterName: string;
  currentValue: string;
  functionName: string;
  position: { line: number; ch: number };
  replacementRange: { start: number; end: number };
}

export interface Variable {
  name: string;
  displayName?: string; // Human-readable name from verbose_call metadata
  type: string;
  value?: any;
  cellId: string;
  compatible: boolean;
  source?: string; // Function or expression that created the variable
}
