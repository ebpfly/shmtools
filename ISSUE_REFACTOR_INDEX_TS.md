# Refactor: Break down 5,466-line monolithic index.ts into modular architecture

## Problem Statement

The JupyterLab extension's main file `/shm_function_selector/src/index.ts` has grown to **5,466 lines**, making it difficult to maintain, test, and understand. For what appears to be a "simple" function selector extension, it's essentially reimplementing an entire UI framework in a single file.

## Current State Analysis

### File Breakdown (5,466 total lines)
- **~1,000 lines**: Inline CSS defined as template literals (`.style.cssText`)
- **426 DOM manipulations**: Direct `createElement`, `innerHTML`, `textContent` calls
- **165 console.log statements**: Debug logging throughout
- **66+ class methods**: All in single `SHMFunctionSelector` class
- **463 comment lines + 438 empty lines**: ~900 lines of comments/whitespace

### Core Issues
1. **No separation of concerns** - CSS, HTML generation, business logic, and utilities all mixed together
2. **Inline everything** - All styles defined as multi-line strings in TypeScript
3. **Manual DOM manipulation** - Building entire UI programmatically without framework
4. **Monolithic class** - Single class handling dropdown, search, keyboard nav, documentation, settings, help, notifications, etc.
5. **Poor testability** - Can't unit test individual components
6. **Difficult debugging** - Hard to find specific functionality in 5,400+ lines

## Proposed Architecture

### Phase 1: Extract Styles (~1,000 lines reduction)
Create `/shm_function_selector/src/styles/` directory:
```
styles/
├── index.css                 # Main stylesheet entry
├── components/
│   ├── dropdown.css          # Dropdown & folding tree styles
│   ├── function-item.css     # Function list item styles
│   ├── documentation.css     # Documentation popup styles
│   ├── notifications.css     # Toast notification styles
│   └── overlay.css           # Modal overlay styles
└── themes/
    ├── light.css            # Light theme variables
    └── dark.css             # Dark theme support
```

### Phase 2: Component Extraction (~2,000 lines reduction)
Create `/shm_function_selector/src/components/` directory:
```
components/
├── FunctionDropdown/
│   ├── index.ts             # Main dropdown component
│   ├── FoldingTree.ts       # Category tree rendering
│   └── SearchBar.ts         # Search functionality
├── DocumentationPanel/
│   ├── index.ts             # Documentation display
│   ├── ParametersSection.ts # Parameter documentation
│   └── ExamplesSection.ts   # Code examples
├── NotificationManager.ts    # Toast notifications
├── SettingsPanel.ts          # Extension settings UI
├── HelpPanel.ts              # Help/tutorial UI
└── ContextMenu.ts            # Right-click context menu
```

### Phase 3: Business Logic Services (~1,000 lines reduction)
Create `/shm_function_selector/src/services/` directory:
```
services/
├── FunctionRegistry.ts       # Function loading & caching
├── CodeGenerator.ts          # Generate code snippets
├── ParameterDetector.ts      # Detect variables in notebook
├── ImportManager.ts          # Manage module imports
├── KeyboardNavigation.ts     # Keyboard shortcut handling
└── NotebookIntegration.ts    # JupyterLab notebook API
```

### Phase 4: Utilities & Types (~500 lines reduction)
Create supporting modules:
```
src/
├── types/
│   ├── index.ts             # Main type exports
│   ├── functions.ts         # SHMFunction interface & related
│   └── ui.ts                # UI component types
├── utils/
│   ├── dom.ts               # DOM manipulation helpers
│   ├── defaults.ts          # Default parameter values
│   └── validation.ts        # Input validation
└── config/
    ├── constants.ts         # App constants
    └── settings.ts          # User settings schema
```

### Phase 5: Main File Refactor
Final `index.ts` structure (~200-300 lines):
```typescript
// index.ts - Clean entry point
import { JupyterFrontEndPlugin } from '@jupyterlab/application';
import { SHMFunctionSelector } from './SHMFunctionSelector';
import { setupKeyboardShortcuts } from './services/KeyboardNavigation';
import './styles/index.css';

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'shm-function-selector:plugin',
  autoStart: true,
  requires: [INotebookTracker, ICommandPalette],
  activate: (app, notebookTracker, palette) => {
    const selector = new SHMFunctionSelector(app, notebookTracker);
    setupKeyboardShortcuts(app, selector);
    // ... minimal setup code
  }
};

export default plugin;
```

## Implementation Strategy

### Step 1: Create Module Structure (No Breaking Changes)
- Create directory structure
- Add index files that re-export from main file
- Ensure build still works

### Step 2: Extract CSS (Low Risk)
- Move inline styles to CSS files
- Use CSS classes instead of inline styles
- Add CSS imports to main file
- Test UI remains unchanged

### Step 3: Extract Pure Functions (Low Risk)
- Move utility functions to utils/
- Move type definitions to types/
- Update imports
- Run tests

### Step 4: Extract Components (Medium Risk)
- Start with simplest components (NotificationManager)
- Move one component at a time
- Test after each extraction
- Keep backward compatibility

### Step 5: Extract Services (Medium Risk)
- Move business logic to services
- Maintain class API surface
- Test thoroughly

### Step 6: Final Cleanup
- Remove console.log statements
- Add proper logging service
- Update documentation
- Add unit tests for modules

## Success Metrics

### Quantitative
- [ ] Main file reduced from 5,466 to <500 lines
- [ ] No single module >500 lines
- [ ] 80%+ code coverage with unit tests
- [ ] Build time unchanged or improved
- [ ] Bundle size reduced by removing duplicated code

### Qualitative
- [ ] Clear separation of concerns
- [ ] Easy to find and modify specific functionality
- [ ] New developers can understand structure quickly
- [ ] Components are reusable and testable
- [ ] Follows TypeScript/React best practices

## Benefits

1. **Maintainability**: Easier to find and fix bugs
2. **Testability**: Can unit test individual components
3. **Reusability**: Components can be reused in other extensions
4. **Performance**: Lazy load components as needed
5. **Collaboration**: Multiple developers can work on different modules
6. **Documentation**: Clear module boundaries make documentation easier
7. **Type Safety**: Better TypeScript inference with smaller modules
8. **Future Features**: Easier to add new functionality

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Incremental refactoring with tests at each step |
| Performance regression | Profile before/after, optimize critical paths |
| Build complexity | Update build config incrementally |
| Lost Git history | Use `git mv` to preserve history |

## Definition of Done

- [ ] All functionality preserved
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Code review completed
- [ ] Performance benchmarked
- [ ] Extension tested in JupyterLab
- [ ] CLAUDE.md updated with new structure

## Priority
**High** - This technical debt is blocking several other improvements:
- Adding new UI features is risky
- Debugging issues takes too long
- New contributors are intimidated by the file size
- Can't implement proper testing

## Labels
`refactoring`, `technical-debt`, `enhancement`, `developer-experience`

## References
- Current file: `/shm_function_selector/src/index.ts`
- Build script: `npm run build:lib && npm run build:labextension:dev`
- Test command: `./restart_jupyterlab.sh`