# Jupyter Notebook SHM Function Plugin - Implementation Plan

## Project Overview
A JupyterLab extension that replicates the Bokeh mFUSE functionality within Jupyter notebooks, enabling streamlined SHM analysis workflows through:
1. **Function Selection Dropdown**: Browse and insert SHM functions by category
2. **Parameter Context Menus**: Right-click parameters to link previous cell outputs
3. **Auto-populated Defaults**: Smart parameter filling with validation

## Architecture Design

### Core Components

#### 1. **SHM Function Registry (`src/registry.ts`)**
- Mirrors Bokeh's function discovery system
- Parses docstring metadata for categories, parameters, GUI specs
- Maintains function catalog with human-readable names
- Caches function signatures and default values

#### 2. **Function Selector Toolbar (`src/toolbar/`)**
```typescript
// Dropdown in notebook toolbar
├── FunctionDropdown.tsx     # Categorized function browser
├── QuickInsert.tsx         # Recently used functions
└── SearchBox.tsx           # Function search/filter
```

#### 3. **Context Menu System (`src/contextmenu/`)**
```typescript
├── ParameterContextMenu.ts  # Right-click on parameters
├── VariableInspector.ts     # Parse available variables
└── CodeParser.ts           # Identify parameter locations
```

#### 4. **Code Generation Engine (`src/codegen/`)**
```typescript
├── TemplateGenerator.ts    # Function call templates
├── ParameterResolver.ts    # Link variables to parameters  
├── DefaultPopulator.ts     # Auto-fill default values
└── ValidationEngine.ts     # Parameter validation
```

#### 5. **Variable Tracking System (`src/variables/`)**
```typescript
├── CellOutputParser.ts     # Extract outputs from cells
├── NamespaceInspector.ts   # Query kernel namespace
├── TypeInference.ts        # Infer compatible variables
└── VariableStore.ts        # Cache variable metadata
```

## Feature Specifications

### 1. Function Selection Dropdown
**Location**: Notebook toolbar (next to cell type selector)
**UI**: Material-UI styled dropdown with search
**Functionality**:
- Categorized function browser (Core, Features, Classification, etc.)
- Search/filter by function name or description
- Recently used functions section
- Insert function call at cursor or new cell

### 2. Parameter Context Menus  
**Trigger**: Right-click on function parameter in code
**Detection**: Parse AST to identify parameter positions
**Menu Options**:
- Available variables (filtered by type compatibility)
- "Use output from Cell X" submenu
- "Set default value" option
- "Show parameter help"

### 3. Auto-population System
**Default Values**: Extract from docstring GUI metadata
**Smart Suggestions**: 
- Match variable types to parameter types
- Suggest most recent compatible outputs
- Validate ranges and constraints

## Implementation Phases

### Phase 1: Foundation (2-3 weeks) ✅ **COMPLETED**
**Core Infrastructure Setup**
- [x] JupyterLab extension scaffolding
- [x] SHM function registry system
- [x] Basic toolbar dropdown for function selection
- [x] Function metadata parsing from docstrings
- [x] Simple code insertion capability

**Deliverables**:
- ✅ Extension installs and loads in JupyterLab
- ✅ Dropdown shows categorized SHM functions (31 functions across 5 categories)
- ✅ Can insert basic function calls into cells

### Phase 2: Variable Tracking (2-3 weeks) ⚠️ **PARTIAL**
**Output Parsing and Variable Management**
- [x] Cell execution monitoring system
- [x] Output parsing for variable extraction
- [x] Basic variable compatibility detection

**Deliverables**:
- ✅ System tracks variables created in notebook cells
- ✅ Can identify available outputs for parameter linking
- ⚠️ **Note**: Integrated into Phase 3 implementation

### Phase 3: Context Menu System (2-3 weeks) ✅ **COMPLETED**
**Right-click Parameter Linking**
- [x] Code parsing to identify parameter positions
- [x] Context menu registration and positioning  
- [x] Variable selection interface
- [x] Code modification for parameter linking
- [x] Smart variable compatibility detection
- [x] Professional menu design with monospace fonts

**Deliverables**:
- ✅ Right-click on parameters shows available variables
- ✅ Can link parameters to previous cell outputs
- ✅ Smart filtering by parameter compatibility
- ✅ Rich visual interface with color coding

### Phase 4: Advanced Features (2-3 weeks) ✅ **COMPLETED**
**Smart Defaults and Validation**
- [x] Auto-population from docstring defaults (GUI metadata parsing)
- [x] Parameter validation system (type checking, range validation)
- [x] Function call templates with placeholders (enhanced code generation)
- [x] Multiple output handling (tuples, dicts) (return info parsing)
- [ ] Undo/redo support for code modifications (pending)

**Deliverables**:
- ✅ Smart parameter auto-completion with GUI widget defaults
- ✅ Validation and error handling with real-time feedback
- ✅ Professional code generation quality with metadata-driven templates
- ✅ Enhanced docstring parsing for GUI metadata, widget specs, and validation rules

### Phase 5: UI Polish and Integration (1-2 weeks)
**User Experience Improvements**
- [ ] Responsive UI design
- [ ] Keyboard shortcuts
- [ ] Function documentation popup
- [ ] Settings panel for preferences
- [ ] Error handling and user feedback

**Deliverables**:
- Production-ready user interface
- Documentation and help system
- Settings and customization options

## Technical Implementation Details

### Function Registry Structure
```typescript
interface SHMFunction {
  name: string;              // Technical name (e.g., "ar_model_shm")
  displayName: string;       // Human readable (e.g., "AR Model Parameters")
  category: string;          // "Features - Time Series Models"
  signature: string;         // Function signature
  parameters: ParameterSpec[];
  returns: ReturnSpec[];
  docstring: string;
  guiMetadata: GUIMetadata;
}

interface ParameterSpec {
  name: string;
  type: string;
  optional: boolean;
  default?: any;
  widget?: WidgetSpec;
  validation?: ValidationRule[];
}
```

### Code Generation Templates
```typescript
// Template for function with auto-populated parameters
const functionTemplate = `
# {{description}}
{{outputs}} = shmtools.{{functionName}}(
{{#parameters}}
    {{name}}={{value}},  # {{description}}
{{/parameters}}
)
`;
```

### Variable Tracking Strategy
```typescript
// Monitor cell execution for variable extraction
notebook.sessionContext.session?.kernel?.executed.connect((sender, args) => {
  const cellId = args.execution_count;
  const outputs = extractOutputVariables(args.content);
  variableStore.updateCell(cellId, outputs);
});
```

## Success Criteria

### Core Functionality
- [ ] Dropdown lists all SHM functions by category
- [ ] Right-click parameter linking works reliably
- [ ] Default values auto-populate from function metadata
- [ ] Generated code executes without syntax errors

### User Experience
- [ ] Intuitive workflow matches Bokeh interface paradigm  
- [ ] Responsive UI performs well in JupyterNotebook
- [ ] Clear visual feedback for parameter sources
- [ ] Robust error handling and recovery

### Integration Quality
- [ ] Compatible with JupyterNotebook
- [ ] Works with existing SHM function docstrings
- [ ] Doesn't interfere with normal notebook operation
- [ ] Extensible for future SHM functions

### Documentation
- [ ] Installation and usage documentation
- [ ] Developer guide for extending functionality
- [ ] Integration guide with SHMTools workflow
- [ ] Example notebooks demonstrating capabilities

This plugin will provide a seamless notebook-native alternative to the Bokeh interface, enabling SHM researchers to build analysis workflows directly within their familiar Jupyter environment while maintaining the guided, form-based approach that makes the tools accessible to domain experts.