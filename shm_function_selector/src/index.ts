import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { INotebookTracker } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';

/**
 * The plugin registration information.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'shm-function-selector:plugin',
  description: 'SHM Function Selector for JupyterLab with context menu parameter linking',
  autoStart: true,
  requires: [INotebookTracker],
  activate: activate
};

/**
 * Activate the SHM Function Selector extension.
 */
function activate(
  app: JupyterFrontEnd,
  notebookTracker: INotebookTracker
): void {
  console.log('🚀 SHM Function Selector JupyterLab extension activated!');

  // Initialize the context menu manager
  const contextMenuManager = new SHMContextMenuManager();

  // Set up basic commands first
  const { commands } = app;
  const commandId = 'shm-selector:show-functions';
  
  commands.addCommand(commandId, {
    label: 'SHM Functions',
    caption: 'SHM Function Selector - Phase 3 Context Menu System',
    execute: () => {
      console.log('📋 SHM Functions command executed');
      alert('✅ SHM Function Selector Phase 3 Active!\n\nImplemented Features:\n\n🎯 Parameter Detection - Right-click on function parameters\n🧠 Smart Variable Compatibility - Recommends matching variables\n✨ Professional Context Menu - Clean interface with type info\n🔧 Code Modification - Links parameters to variables automatically\n\n➡️ Try right-clicking on parameter values in code cells!');
    }
  });

  // Set up notebook tracking with full context menu functionality
  notebookTracker.widgetAdded.connect((sender, nbPanel) => {
    console.log('📓 Notebook added, setting up SHM context menu functionality');
    
    const notebook = nbPanel.content;
    
    // Add a toolbar button to the notebook
    const button = document.createElement('button');
    button.textContent = '🔗 SHM Parameter Linker';
    button.style.cssText = `
      margin: 5px;
      padding: 5px 10px;
      background: #2e7d2e;
      color: white;
      border: none;
      border-radius: 3px;
      cursor: pointer;
      font-size: 12px;
      font-weight: bold;
    `;
    button.onclick = () => {
      app.commands.execute(commandId);
    };
    
    // Add button to notebook toolbar
    const toolbar = nbPanel.toolbar;
    if (toolbar) {
      toolbar.node.appendChild(button);
      console.log('🔧 Added SHM parameter linker button to notebook toolbar');
    }
    
    // Listen for right-click events on code cells with full functionality
    notebook.node.addEventListener('contextmenu', (event: MouseEvent) => {
      const activeCell = notebook.activeCell;
      if (!activeCell || activeCell.model.type !== 'code') {
        return;
      }

      // Get cursor position and code content
      const editor = activeCell.editor;
      if (!editor) return;

      const cursor = editor.getCursorPosition();
      const code = editor.model.sharedModel.getSource();
      
      // Get the index of the current cell
      const currentCellIndex = notebook.widgets.indexOf(activeCell);
      
      // Calculate absolute cursor position in text
      const lines = code.split('\n');
      let absolutePos = 0;
      for (let i = 0; i < cursor.line; i++) {
        absolutePos += lines[i].length + 1; // +1 for newline
      }
      absolutePos += cursor.column;

      console.log('🎯 Right-click at position:', cursor, 'absolute:', absolutePos);
      console.log('📍 Current cell index:', currentCellIndex);
      
      // Try to detect parameter context
      const parameterContext = contextMenuManager.detectParameterContext(code, absolutePos);
      
      if (parameterContext) {
        console.log('🎯 Parameter detected:', parameterContext);
        
        // Prevent default context menu
        event.preventDefault();
        event.stopPropagation();
        
        // Show SHM context menu with current cell index
        contextMenuManager.showContextMenu(event, parameterContext, notebook, currentCellIndex);
      } else {
        console.log('📝 No parameter detected at cursor position');
        
        // Show a brief indicator that the system is working but no parameter found
        const notification = document.createElement('div');
        notification.textContent = '🔍 Position cursor on a parameter value and right-click';
        notification.style.cssText = `
          position: fixed;
          top: 20px;
          right: 20px;
          background: #ff9800;
          color: white;
          padding: 8px 12px;
          border-radius: 4px;
          z-index: 10000;
          font-family: monospace;
          font-size: 11px;
        `;
        document.body.appendChild(notification);
        
        setTimeout(() => {
          if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
          }
        }, 2000);
      }
    });
  });

  console.log('✅ SHM Function Selector Phase 3 setup complete - Context Menu System loaded!');
}

export default plugin;

// ============================================================================
// PHASE 3: CONTEXT MENU IMPLEMENTATION
// ============================================================================

interface ParameterContext {
  parameterName: string;
  currentValue: string;
  functionName: string;
  position: { line: number; ch: number };
  replacementRange: { start: number; end: number };
}

interface Variable {
  name: string;
  type: string;
  value?: any;
  cellId: string;
  compatible: boolean;
  source?: string; // Function or expression that created the variable
}

class SHMContextMenuManager {
  private variables: Variable[] = [];
  private contextMenu: HTMLElement | null = null;

  /**
   * Parse code to detect parameter context at cursor position
   */
  detectParameterContext(code: string, cursorPos: number): ParameterContext | null {
    console.log(`🔍 Full code length: ${code.length}, cursor at: ${cursorPos}`);
    console.log(`🔍 Code around cursor: "${code.substring(Math.max(0, cursorPos-10), cursorPos+10)}"`);
    
    const lines = code.split('\n');
    let currentPos = 0;
    let targetLine = 0;
    let targetCol = 0;

    // Find line and column of cursor
    for (let i = 0; i < lines.length; i++) {
      if (currentPos + lines[i].length >= cursorPos) {
        targetLine = i;
        targetCol = cursorPos - currentPos;
        break;
      }
      currentPos += lines[i].length + 1; // +1 for newline
    }

    const line = lines[targetLine];
    
    console.log(`🔍 Target line ${targetLine}: "${line}"`);
    console.log(`🔍 Target column: ${targetCol}`);
    
    // Simple character-by-character approach to find parameter
    return this.findParameterAtPosition(line, targetCol, targetLine, currentPos);
  }

  private findParameterAtPosition(line: string, col: number, lineNum: number, lineStartPos: number): ParameterContext | null {
    console.log(`🔍 Character at cursor (${col}): "${line[col]}"`);
    
    // For line "d,e = func( g=None, h=None)", find the function call portion
    const functionCallMatch = line.match(/(\w+)\s*\(([^)]+)\)/);
    if (!functionCallMatch) {
      console.log(`🔍 No function call found in line`);
      return null;
    }
    
    const functionName = functionCallMatch[1];
    const parameterList = functionCallMatch[2];
    const functionStart = functionCallMatch.index!;
    const parenStart = line.indexOf('(', functionStart);
    
    console.log(`🔍 Function "${functionName}" with parameters: "${parameterList}"`);
    console.log(`🔍 Function parentheses start at: ${parenStart}`);
    
    // Check if cursor is within the function call parentheses
    if (col < parenStart || col > parenStart + parameterList.length + 1) {
      console.log(`🔍 Cursor not within function call parentheses`);
      return null;
    }
    
    // Parse individual parameters in the list
    const parameters = [];
    const paramPattern = /(\w+)\s*=\s*([^,]+)/g;
    let match;
    
    while ((match = paramPattern.exec(parameterList)) !== null) {
      const paramName = match[1];
      const paramValue = match[2].trim();
      
      // Calculate absolute positions
      const paramStart = parenStart + 1 + match.index;
      const valueStart = paramStart + paramName.length + 1; // +1 for '='
      while (valueStart < line.length && /\s/.test(line[valueStart])) {
        // Skip whitespace after =
      }
      const valueEnd = parenStart + 1 + match.index + match[0].length - 1;
      
      parameters.push({
        name: paramName,
        value: paramValue,
        valueStart: valueStart,
        valueEnd: valueEnd
      });
      
      console.log(`🔍 Parameter "${paramName}" = "${paramValue}" at [${valueStart}-${valueEnd}]`);
    }
    
    // Find which parameter the cursor is in
    for (const param of parameters) {
      console.log(`🔍 Checking if cursor ${col} is in "${param.name}" range [${param.valueStart}-${param.valueEnd}]`);
      
      // Check if cursor is within the parameter value OR just after it (on comma/space)
      if (col >= param.valueStart && col <= param.valueEnd + 2) {
        console.log(`✅ Found parameter "${param.name}" at cursor position`);
        
        return {
          parameterName: param.name,
          currentValue: param.value,
          functionName: functionName,
          position: { line: lineNum, ch: col },
          replacementRange: { 
            start: lineStartPos - line.length + param.valueStart,
            end: lineStartPos - line.length + param.valueEnd + 1
          }
        };
      }
    }
    
    console.log(`🔍 Cursor not in any parameter value`);
    return null;
  }

  /**
   * Extract variables from notebook cells
   */
  extractVariablesFromCells(notebook: any): void {
    this.variables = [];
    
    for (let i = 0; i < notebook.model.cells.length; i++) {
      const cell = notebook.model.cells.get(i);
      if (cell.type === 'code') {
        // Use sharedModel.getSource() instead of value.text
        const cellCode = cell.sharedModel.getSource();
        const cellId = `cell-${i}`;
        
        // Simple variable extraction patterns
        this.extractVariablesFromCode(cellCode, cellId);
      }
    }
  }

  private extractVariablesFromCode(code: string, cellId: string): void {
    // Pattern 1: Simple assignment: variable = expression
    const simpleAssignPattern = /^(\w+)\s*=\s*(.+)$/gm;
    let match;
    
    while ((match = simpleAssignPattern.exec(code)) !== null) {
      const varName = match[1];
      const expression = match[2].trim();
      
      // Skip common non-data variables
      if (['i', 'j', 'k', 'n', 'len', 'idx'].includes(varName)) {
        continue;
      }

      // Extract function name if it's a function call
      const funcMatch = expression.match(/^(\w+)\s*\(/);
      const source = funcMatch ? funcMatch[1] + '()' : expression.substring(0, 30) + (expression.length > 30 ? '...' : '');

      const variable: Variable = {
        name: varName,
        type: this.inferVariableType(code, varName),
        cellId: cellId,
        compatible: false, // Will be set based on parameter context
        source: source
      };

      this.variables.push(variable);
    }
    
    // Pattern 2: Tuple unpacking: a, b = func() or a,b,c = values
    const tuplePattern = /^([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)+)\s*=\s*(.+)$/gm;
    
    while ((match = tuplePattern.exec(code)) !== null) {
      const varsString = match[1];
      const expression = match[2].trim();
      const varNames = varsString.split(',').map(v => v.trim());
      
      // Extract function name if it's a function call
      const funcMatch = expression.match(/^(\w+)\s*\(/);
      const source = funcMatch ? funcMatch[1] + '()' : expression.substring(0, 30) + (expression.length > 30 ? '...' : '');
      
      for (const varName of varNames) {
        if (varName && !['i', 'j', 'k', 'n', 'len', 'idx'].includes(varName)) {
          const variable: Variable = {
            name: varName,
            type: 'unknown', // Hard to infer type from tuple unpacking
            cellId: cellId,
            compatible: false,
            source: source
          };
          
          this.variables.push(variable);
        }
      }
    }
    
    // Pattern 3: Import statements that create variables
    const importPattern = /import\s+(\w+)|from\s+\w+\s+import\s+(\w+)/g;
    
    while ((match = importPattern.exec(code)) !== null) {
      const varName = match[1] || match[2];
      if (varName && varName !== 'as') {
        const variable: Variable = {
          name: varName,
          type: 'module',
          cellId: cellId,
          compatible: false
        };
        
        this.variables.push(variable);
      }
    }
    
    console.log(`📊 Extracted ${this.variables.filter(v => v.cellId === cellId).length} variables from ${cellId}`);
  }

  private inferVariableType(code: string, varName: string): string {
    // Simple type inference heuristics
    if (code.includes(`${varName} = np.`) || code.includes(`${varName} = numpy.`)) {
      return 'numpy.ndarray';
    }
    if (code.includes(`${varName} = pd.`) || code.includes(`${varName} = pandas.`)) {
      return 'pandas.DataFrame';
    }
    if (code.includes(`${varName} = [`)) {
      return 'list';
    }
    if (code.includes(`${varName} = {`)) {
      return 'dict';
    }
    
    return 'unknown';
  }

  /**
   * Determine variable compatibility with parameter
   */
  isVariableCompatible(variable: Variable, parameterContext: ParameterContext): boolean {
    const paramName = parameterContext.parameterName.toLowerCase();
    
    // Data parameters expect arrays
    if (['data', 'input_data', 'features', 'signals'].includes(paramName)) {
      return ['numpy.ndarray', 'pandas.DataFrame', 'list'].includes(variable.type);
    }
    
    // Frequency parameters expect numbers
    if (['fs', 'sampling_rate', 'freq', 'frequency'].includes(paramName)) {
      return variable.name.toLowerCase().includes('freq') || 
             variable.name.toLowerCase().includes('fs') ||
             variable.name.toLowerCase().includes('rate');
    }
    
    // Order parameters expect integers
    if (['order', 'n_components', 'n_features'].includes(paramName)) {
      return variable.name.toLowerCase().includes('order') ||
             variable.name.toLowerCase().includes('n_') ||
             !isNaN(parseInt(variable.name));
    }
    
    return true; // Default to compatible
  }

  /**
   * Show context menu with available variables
   */
  showContextMenu(
    event: MouseEvent, 
    parameterContext: ParameterContext, 
    notebook: any,
    currentCellIndex: number = -1
  ): void {
    this.hideContextMenu();
    
    // Extract variables from all cells
    this.extractVariablesFromCells(notebook);
    
    // Filter to only show variables from cells before the current one
    if (currentCellIndex >= 0) {
      this.variables = this.variables.filter(v => {
        const cellNumber = parseInt(v.cellId.replace('cell-', ''));
        return cellNumber < currentCellIndex;
      });
    }
    
    console.log(`📊 Variables from cells before cell ${currentCellIndex}: ${this.variables.length}`);
    this.variables.forEach(v => {
      console.log(`  - ${v.name} (${v.type}) from ${v.source || 'unknown'} in ${v.cellId}`);
    });
    
    // Filter and sort variables by compatibility
    const compatibleVars = this.variables.filter(v => 
      this.isVariableCompatible(v, parameterContext)
    );
    const otherVars = this.variables.filter(v => 
      !this.isVariableCompatible(v, parameterContext)
    );

    // Create context menu
    this.contextMenu = document.createElement('div');
    this.contextMenu.className = 'shm-context-menu';
    this.contextMenu.style.cssText = `
      position: fixed;
      left: ${event.pageX}px;
      top: ${event.pageY}px;
      background: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 12px;
      z-index: 10000;
      max-height: 300px;
      overflow-y: auto;
      min-width: 200px;
    `;

    // Add header
    const header = document.createElement('div');
    header.textContent = `Link parameter: ${parameterContext.parameterName}`;
    header.style.cssText = `
      padding: 8px 12px;
      background: #f5f5f5;
      border-bottom: 1px solid #ddd;
      font-weight: bold;
      color: #333;
    `;
    this.contextMenu.appendChild(header);

    // Add compatible variables section
    if (compatibleVars.length > 0) {
      const compatibleHeader = document.createElement('div');
      compatibleHeader.textContent = 'Recommended';
      compatibleHeader.style.cssText = `
        padding: 6px 12px;
        background: #e8f5e8;
        color: #2e7d2e;
        font-weight: bold;
        border-bottom: 1px solid #ddd;
      `;
      this.contextMenu.appendChild(compatibleHeader);

      compatibleVars.forEach(variable => {
        this.addVariableMenuItem(variable, parameterContext, notebook, true);
      });
    }

    // Add other variables section
    if (otherVars.length > 0) {
      const otherHeader = document.createElement('div');
      otherHeader.textContent = 'Other variables';
      otherHeader.style.cssText = `
        padding: 6px 12px;
        background: #f0f0f0;
        color: #666;
        font-weight: bold;
        border-bottom: 1px solid #ddd;
      `;
      this.contextMenu.appendChild(otherHeader);

      otherVars.forEach(variable => {
        this.addVariableMenuItem(variable, parameterContext, notebook, false);
      });
    }

    if (compatibleVars.length === 0 && otherVars.length === 0) {
      const noVars = document.createElement('div');
      noVars.textContent = 'No variables found';
      noVars.style.cssText = `
        padding: 12px;
        color: #999;
        font-style: italic;
      `;
      this.contextMenu.appendChild(noVars);
    }

    document.body.appendChild(this.contextMenu);

    // Close menu on outside click
    const closeHandler = (e: MouseEvent) => {
      if (!this.contextMenu?.contains(e.target as Node)) {
        this.hideContextMenu();
        document.removeEventListener('click', closeHandler);
      }
    };
    
    setTimeout(() => {
      document.addEventListener('click', closeHandler);
    }, 100);
  }

  private addVariableMenuItem(
    variable: Variable, 
    parameterContext: ParameterContext, 
    notebook: any,
    isRecommended: boolean
  ): void {
    const menuItem = document.createElement('div');
    menuItem.className = 'shm-context-menu-item';
    menuItem.style.cssText = `
      padding: 8px 12px;
      cursor: pointer;
      border-bottom: 1px solid #eee;
      transition: background 0.2s;
      ${isRecommended ? 'background: #f0fff0;' : ''}
    `;

    menuItem.innerHTML = `
      <div style="font-weight: bold; color: ${isRecommended ? '#2e7d2e' : '#333'};">
        ${variable.name}
      </div>
      <div style="font-size: 10px; color: #666;">
        ${variable.source ? `from ${variable.source} • ` : ''}${variable.type} • ${variable.cellId}
      </div>
    `;

    menuItem.addEventListener('mouseenter', () => {
      menuItem.style.background = isRecommended ? '#e8f5e8' : '#f5f5f5';
    });

    menuItem.addEventListener('mouseleave', () => {
      menuItem.style.background = isRecommended ? '#f0fff0' : 'white';
    });

    menuItem.addEventListener('click', () => {
      this.linkParameterToVariable(variable, parameterContext, notebook);
      this.hideContextMenu();
    });

    this.contextMenu!.appendChild(menuItem);
  }

  /**
   * Replace parameter value with selected variable
   */
  linkParameterToVariable(
    variable: Variable, 
    parameterContext: ParameterContext, 
    notebook: any
  ): void {
    const activeCell = notebook.activeCell;
    if (!activeCell) return;

    const editor = activeCell.editor;
    const currentText = editor.model.sharedModel.getSource();
    
    // Simple replacement: replace current parameter value with variable name
    const lines = currentText.split('\n');
    const targetLine = parameterContext.position.line;
    
    if (targetLine < lines.length) {
      let line = lines[targetLine];
      
      // Replace the parameter value
      const paramPattern = new RegExp(`(${parameterContext.parameterName}\\s*=\\s*)([^,\\)]+)`, 'g');
      line = line.replace(paramPattern, `$1${variable.name}`);
      
      // Remove TODO comments
      line = line.replace(/\s*#\s*TODO[^\n]*/g, '');
      
      lines[targetLine] = line;
      
      // Update the cell content
      editor.model.sharedModel.setSource(lines.join('\n'));
    }

    // Show success notification
    this.showNotification(`✅ Linked ${parameterContext.parameterName} = ${variable.name}`, '#4caf50');
  }

  private showNotification(message: string, color: string): void {
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: ${color};
      color: white;
      padding: 10px 15px;
      border-radius: 4px;
      z-index: 10000;
      font-family: monospace;
      font-size: 12px;
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 3000);
  }

  hideContextMenu(): void {
    if (this.contextMenu && this.contextMenu.parentNode) {
      this.contextMenu.parentNode.removeChild(this.contextMenu);
    }
    this.contextMenu = null;
  }
}