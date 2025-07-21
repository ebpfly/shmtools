# mFUSE to Bokeh Conversion Plan

## Current State Assessment

**✅ What's Working:**
- Complete 4-panel UI layout established
- Function discovery and categorization framework
- Basic workflow step management  
- Parameter widget generation foundation
- Results viewer with plotting capabilities

**❌ Critical Gaps:**
- No inter-panel communication
- Workflow execution is placeholder-only
- Parameter forms hardcoded vs dynamic
- No session file compatibility
- Missing data flow between steps

---

## Phase 1: Core Integration (2-3 weeks)
*Make the existing panels work together as a functional workflow builder*

### Week 1: Panel Communication
1. **Function Library → Workflow Builder**
   - Add function to workflow sequence via double-click/button
   - Pass function metadata to workflow state
   - Update workflow step table with new entries

2. **Workflow Builder → Parameter Controls**
   - Select workflow step to configure parameters  
   - Load parameter specifications for selected function
   - Display current parameter values with proper widgets

3. **Parameter Controls → Workflow Builder**
   - Apply parameter changes to selected workflow step
   - Update step status (configured/ready/error)
   - Validate parameters and show error states

### Week 2: Basic Workflow Execution
1. **Real Function Invocation**
   - Connect to actual SHMTools functions (not hardcoded examples)
   - Execute single steps with real parameters
   - Capture outputs and error handling

2. **Data Flow Management**
   - Simple variable workspace for passing data between steps
   - Output → Input connections for sequential workflows
   - Basic data type checking and validation

### Week 3: Parameter Introspection
1. **Dynamic Parameter Form Generation**
   - Extract parameter specs from SHMTools docstrings automatically
   - Generate appropriate widgets based on docstring metadata
   - Remove hardcoded parameter specifications

2. **Function Integration Testing**
   - Test with Phase 1 example functions (`arModel_shm`, `learnPCA_shm`, etc.)
   - Validate parameter types and function execution
   - Ensure outputs can be visualized in Results panel

---

## Phase 2: mFUSE Feature Parity (3-4 weeks)
*Implement core mFUSE features for workflow building and management*

### Session File Compatibility
1. **mFUSE .ses Import (Week 4)**
   - Parse mFUSE session file format
   - Convert steps to Bokeh workflow format
   - Map parameter specifications and user inputs
   - Handle QuickStep code blocks

2. **Session Management (Week 5)**
   - Save/load Bokeh native session format
   - Maintain workflow state persistence
   - Export workflows to executable Python scripts

### Enhanced Workflow Features (Weeks 6-7)
1. **Advanced Data Flow**
   - Variable tagging and output marking
   - Input/output connections with visual indicators
   - Support for multiple function call methods
   - User input vs auto-derived parameter handling

2. **Workflow Execution Engine**
   - Full sequential workflow execution
   - Progress tracking and user feedback
   - Error handling with step-level recovery
   - Result caching and re-execution optimization

---

## Phase 3: Enhanced User Experience (2-3 weeks)
*Polish the interface and add advanced features*

### Improved Visualization (Week 8)
1. **Dynamic Plot Generation**
   - Auto-generate plots based on function outputs
   - Support all plot types from original examples
   - Interactive plot tools and annotations

2. **Multi-Plot Layouts**
   - Comparison views for outlier detection algorithms
   - Dashboard-style result summaries
   - Export capabilities for publications

### Data Management (Week 9)
1. **File Upload/Import**
   - Support for .mat, .csv, .npy data formats
   - Dataset preview and validation
   - Integration with example data loading

2. **Variable Workspace**
   - MATLAB-like variable browser
   - Data inspection and preview capabilities
   - Manual variable creation and editing

### Advanced Features (Week 10)
1. **QuickStep Code Execution**
   - Inline Python code blocks in workflows
   - Integration with workflow variable space
   - Code editing with syntax highlighting

2. **Workflow Optimization**
   - Step reordering with dependency checking
   - Parallel execution where possible
   - Performance monitoring and optimization hints

---

## Implementation Strategy

### Development Approach
1. **Iterative Integration**: Get basic functionality working before adding features
2. **Example-Driven Testing**: Use Phase 1 PCA example as primary test case
3. **Backward Compatibility**: Ensure mFUSE workflows can be imported
4. **Progressive Enhancement**: Start with core workflows, add advanced features incrementally

### Testing Strategy
1. **Unit Tests**: Each panel component individually
2. **Integration Tests**: Cross-panel communication
3. **Workflow Tests**: End-to-end example execution
4. **Migration Tests**: mFUSE session file imports

### Key Technical Decisions
1. **Session Format**: JSON-based for Bokeh native, XML parser for mFUSE compatibility
2. **Variable Storage**: In-memory workspace with optional persistence
3. **Function Discovery**: Docstring-based with fallback to manual specifications
4. **Plot Integration**: Bokeh native plots with matplotlib fallback for complex visualizations

---

## Success Criteria

### Phase 1 Success
- User can build and execute simple PCA outlier detection workflow
- All panels communicate properly
- Real SHMTools functions execute with correct parameters

### Phase 2 Success
- Existing mFUSE workflows can be imported and executed
- All core workflow features functional
- Session management working

### Phase 3 Success
- Feature parity with original mFUSE
- Enhanced visualization and data management
- Ready for production use with example-driven development

---

## mFUSE Architecture Analysis

### 4-Panel Layout Structure
mFUSE uses a **4-panel layout** divided into two main sections:

**Top Section (Main Workspace):**
1. **Function Library Panel** (Left) - Hierarchical tree view of available functions
2. **Sequence Panel** (Center-Right) - List of workflow steps
3. **Step Panel** (Right) - Parameter configuration for selected step

**Bottom Section (Tabbed Interface):**
4. **Three-Tab Panel** with:
   - **Function Documentation Tab** - Shows help/documentation for selected functions
   - **Sequence M-File Tab** - Generated MATLAB code preview and export
   - **User Inputs/Tagged Outputs Tab** - Parameter tweaking and results viewing

### Key mFUSE Features
1. **Function Library Panel**
   - Tree-based function browser organized by categories/directories
   - Drag-and-drop functionality to add functions to sequences
   - Dynamic loading from directories specified in `mFUSElibrary.txt`
   - Search and filtering capabilities

2. **Sequence Panel**
   - Linear workflow builder showing step-by-step execution order
   - Drag-and-drop reordering of sequence steps
   - Step selection for parameter editing
   - Visual step status indicators (ready, error, etc.)
   - Quick Steps (QS) - Custom MATLAB code snippets

3. **Step Panel**
   - Dynamic parameter forms based on function signatures
   - Input/output variable mapping between steps
   - Call method selection (different function overloads)
   - Parameter validation with visual status indicators

4. **User Inputs/Tagged Outputs Panel**
   - Global parameter table showing all user inputs across workflow
   - Tagged outputs viewer for results inspection
   - Plotting capabilities (line, bar, point, semilog plots)
   - Data export/save functionality

### Session Files (.ses Format)
- XML-like text format storing complete workflow definitions
- Step definitions with function calls, parameters, and connections
- User inputs and parameter values
- Output tagging information
- Metadata (version, author, descriptions)

### Workflow Creation Process
1. **Browse Function Library** - Navigate categorized function tree
2. **Add Functions to Sequence** - Drag-and-drop or double-click
3. **Configure Parameters** - Set inputs in Step Panel
4. **Connect Data Flow** - Map outputs to inputs of subsequent steps
5. **Execute/Preview** - Generate and run MATLAB code
6. **Review Results** - View outputs and plots in Tagged Outputs tab

---

## Current Bokeh Implementation Status

### What's Already Implemented ✅

#### Core Application Structure
- **Entry Point**: `bokeh_shmtools/app.py`
- **Panel System**: All 4 panels are structurally implemented
- **Layout Management**: Proper Bokeh layouts with responsive sizing
- **Modular Design**: Clean separation of concerns across panels

#### Function Library Panel (Partially Functional)
- **Category Browser**: Dropdown with predefined SHM function categories
- **Function Discovery**: Dynamic discovery of SHMTools functions via introspection  
- **Function Table**: Displays function names and descriptions
- **Docstring Parser**: Sophisticated parser for extracting GUI metadata from function docstrings

#### Workflow Builder Panel (Partially Functional)
- **Step Management**: Add, remove, reorder workflow steps
- **Workflow State**: Tracks step parameters, status, and outputs
- **Visual Workflow**: Table showing step sequence with status indicators
- **Control Buttons**: Move up/down, remove, clear all functionality

#### Parameter Controls Panel (Partially Functional)
- **Dynamic GUI Generation**: Creates appropriate widgets based on parameter types
- **Widget Types**: NumericInput, Select, Checkbox, TextInput
- **Parameter Validation**: Min/max constraints and type checking
- **Apply/Reset**: Parameter change management

#### Results Viewer Panel (Partially Functional)
- **Multi-Tab Interface**: Plots, Data, Export tabs
- **Interactive Plotting**: Bokeh plots with pan/zoom/reset tools
- **Plot Types**: Time series, PSD, spectrogram, outlier scores
- **Data Tables**: Tabular display of analysis results

### Critical Gaps ❌

1. **Function Discovery Integration**
   - Function discovery fails gracefully but falls back to hardcoded examples
   - No real connection between discovered functions and parameter specifications

2. **Inter-Panel Communication** 
   - Panels operate independently with TODO comments for integration
   - Missing workflows between all panel combinations

3. **Workflow Execution Engine**
   - Placeholder implementation that just prints status
   - No function invocation, parameter validation, or data flow

4. **Session File Compatibility**
   - Cannot import/export mFUSE `.ses` session files
   - No migration path for existing workflows

5. **Parameter Introspection**
   - Hardcoded parameter specs for example functions only
   - No dynamic extraction from SHMTools function signatures

6. **Data Management**
   - No file upload/import capabilities
   - No data passing between workflow steps
   - No variable workspace management

This conversion plan transforms the existing Bokeh foundation into a fully functional replacement for mFUSE while maintaining compatibility and extending capabilities for modern web-based usage.