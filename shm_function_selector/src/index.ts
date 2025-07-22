import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { INotebookTracker } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { requestAPI } from './serverAPI';

/**
 * The plugin registration information.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'shm-function-selector:plugin',
  description: 'SHM Function Selector for JupyterLab with function dropdown and context menu parameter linking',
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

  // Initialize managers
  const contextMenuManager = new SHMContextMenuManager();
  const functionSelector = new SHMFunctionSelector(app, notebookTracker);

  // Set up basic commands first
  const { commands } = app;
  const commandId = 'shm-selector:show-functions';
  
  commands.addCommand(commandId, {
    label: 'SHM Functions',
    caption: 'SHM Function Selector - Browse and insert SHM functions',
    execute: () => {
      console.log('📋 SHM Functions command executed');
      alert('✅ SHM Function Selector Active!\n\nFeatures:\n\n📚 Function Browser - Click dropdown to browse categorized functions\n🎯 Parameter Detection - Right-click on function parameters\n🧠 Smart Variable Compatibility - Recommends matching variables\n✨ Professional Context Menu - Clean interface with type info\n🔧 Code Modification - Links parameters to variables automatically\n\n➡️ Try the function dropdown or right-click on parameter values!');
    }
  });

  // Set up notebook tracking with full context menu functionality
  notebookTracker.widgetAdded.connect((sender, nbPanel) => {
    console.log('📓 Notebook added, setting up SHM context menu functionality');
    
    const notebook = nbPanel.content;
    
    // Note: Removed the red SHM Parameter Linker button per user request
    
    // Listen for right-click events on code cells with full functionality
    notebook.node.addEventListener('contextmenu', (event: MouseEvent) => {
      const activeCell = notebook.activeCell;
      if (!activeCell || activeCell.model.type !== 'code') {
        return;
      }

      // Get cursor position and code content
      const editor = activeCell.editor;
      if (!editor) return;

      // Clear any text selection to get accurate cursor position
      const selection = editor.getSelection();
      let cursor;
      
      if (selection && selection.start.line === selection.end.line && selection.start.column === selection.end.column) {
        // No selection, use cursor position
        cursor = editor.getCursorPosition();
      } else if (selection) {
        // Text is selected, use the start of selection as cursor position
        cursor = selection.start;
        console.log('🔍 Text selected, using selection start as cursor position');
      } else {
        // Fallback to cursor position
        cursor = editor.getCursorPosition();
      }

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
      console.log('📋 Full code:');
      console.log(code);
      console.log('📋 Code length:', code.length);
      console.log('📋 Character at absolute position:', code[absolutePos] || 'END');
      console.log('📋 Context around cursor:', code.substring(Math.max(0, absolutePos-10), absolutePos+10));
      
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

  console.log('✅ SHM Function Selector setup complete - Function Dropdown (Phase 1) and Context Menu (Phase 3) loaded!');
}

export default plugin;

// ============================================================================
// PHASE 1: FUNCTION SELECTOR DROPDOWN IMPLEMENTATION
// ============================================================================

interface SHMFunction {
  name: string;
  displayName: string;
  category: string;
  signature: string;
  description: string;
  docstring: string;
  parameters: Array<{
    name: string;
    type: string;
    optional: boolean;
    default: string | null;
    description?: string;
  }>;
}

class SHMFunctionSelector {
  private app: JupyterFrontEnd;
  private notebookTracker: INotebookTracker;
  private functions: SHMFunction[] = [];
  private dropdown: HTMLSelectElement | null = null;
  private recentlyUsed: string[] = [];

  constructor(app: JupyterFrontEnd, notebookTracker: INotebookTracker) {
    this.app = app;
    this.notebookTracker = notebookTracker;
    
    // Load functions from server
    this.loadFunctions();
    
    // Set up notebook tracking
    this.setupNotebookTracking();
  }

  private async loadFunctions(): Promise<void> {
    try {
      console.log('📥 Loading SHM functions from server...');
      const functions = await requestAPI<SHMFunction[]>('functions');
      this.functions = functions;
      console.log(`✅ Loaded ${functions.length} SHM functions`, functions);
      
      // If dropdown exists, populate it
      if (this.dropdown) {
        this.populateDropdown();
      }
    } catch (error) {
      console.error('❌ Failed to load SHM functions:', error);
      console.error('Error details:', error);
      
      // Show error notification
      this.showNotification('⚠️ Failed to load SHM functions. Check browser console.', '#ff9800');
    }
  }

  private setupNotebookTracking(): void {
    this.notebookTracker.widgetAdded.connect((sender, nbPanel) => {
      console.log('📓 Adding function selector to notebook toolbar');
      
      // Create the dropdown container
      const container = document.createElement('div');
      container.style.cssText = `
        display: inline-flex;
        align-items: center;
        margin: 2px 5px;
        gap: 5px;
        flex-shrink: 0;
        white-space: nowrap;
        z-index: 1000;
      `;

      // Create label
      const label = document.createElement('label');
      label.textContent = 'SHM Function:';
      label.style.cssText = `
        font-size: 12px;
        font-weight: bold;
        color: #333;
      `;

      // Create dropdown
      this.dropdown = document.createElement('select');
      this.dropdown.style.cssText = `
        padding: 4px 8px;
        font-size: 11px;
        border: 1px solid #ccc;
        border-radius: 3px;
        background: white;
        cursor: pointer;
        min-width: 180px;
        max-width: 250px;
        flex-shrink: 1;
      `;

      // Add automatic insertion on selection change
      this.dropdown.onchange = () => this.insertSelectedFunction(nbPanel);

      // Add elements to container
      container.appendChild(label);
      container.appendChild(this.dropdown);

      // Add to notebook toolbar
      const toolbar = nbPanel.toolbar;
      if (toolbar) {
        toolbar.node.appendChild(container);
        console.log('✅ Added function selector to toolbar');
        
        // Populate dropdown
        this.populateDropdown();
      }
    });
  }

  private populateDropdown(): void {
    if (!this.dropdown) return;

    // Clear existing options
    this.dropdown.innerHTML = '';

    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select a function --';
    this.dropdown.appendChild(defaultOption);

    // Add recently used section if any
    if (this.recentlyUsed.length > 0) {
      const recentGroup = document.createElement('optgroup');
      recentGroup.label = '⏱️ Recently Used';
      
      this.recentlyUsed.forEach(funcName => {
        const func = this.functions.find(f => f.name === funcName);
        if (func) {
          const option = document.createElement('option');
          option.value = func.name;
          option.textContent = func.displayName;
          recentGroup.appendChild(option);
        }
      });
      
      this.dropdown.appendChild(recentGroup);
    }

    // Group functions by category
    const categories = new Map<string, SHMFunction[]>();
    this.functions.forEach(func => {
      if (!categories.has(func.category)) {
        categories.set(func.category, []);
      }
      categories.get(func.category)!.push(func);
    });

    // Add categorized functions
    categories.forEach((funcs, category) => {
      const optgroup = document.createElement('optgroup');
      optgroup.label = category;
      
      funcs.forEach(func => {
        const option = document.createElement('option');
        option.value = func.name;
        option.textContent = func.displayName;
        option.title = func.description;
        optgroup.appendChild(option);
      });
      
      this.dropdown.appendChild(optgroup);
    });
  }

  private insertSelectedFunction(nbPanel: any): void {
    if (!this.dropdown || !this.dropdown.value) {
      // Don't show alert for automatic calls - just return silently
      return;
    }

    const selectedFunc = this.functions.find(f => f.name === this.dropdown!.value);
    if (!selectedFunc) return;

    // Add to recently used (max 5)
    this.recentlyUsed = [selectedFunc.name, ...this.recentlyUsed.filter(n => n !== selectedFunc.name)].slice(0, 5);
    
    // Generate code snippet
    const codeSnippet = this.generateCodeSnippet(selectedFunc);
    
    // Insert into active cell or create new cell
    const notebook = nbPanel.content;
    const activeCell = notebook.activeCell;
    
    if (activeCell && activeCell.model.type === 'code') {
      // Insert at cursor position in active cell
      const editor = activeCell.editor;
      if (editor) {
        const cursorPos = editor.getCursorPosition();
        const currentText = editor.model.sharedModel.getSource();
        
        // Insert code at cursor position
        const lines = currentText.split('\n');
        const line = lines[cursorPos.line] || '';
        const before = line.substring(0, cursorPos.column);
        const after = line.substring(cursorPos.column);
        
        // If we're in the middle of a line, add newlines
        const insertion = (before.trim() ? '\n' : '') + codeSnippet + (after.trim() ? '\n' : '');
        lines[cursorPos.line] = before + insertion + after;
        
        editor.model.sharedModel.setSource(lines.join('\n'));
        
        // Move cursor to first parameter
        const newCursorLine = cursorPos.line + (before.trim() ? 1 : 0);
        editor.setCursorPosition({ line: newCursorLine, column: codeSnippet.indexOf('=') + 1 });
      }
    } else {
      // Create new code cell
      const cellIndex = notebook.activeCellIndex !== -1 ? notebook.activeCellIndex + 1 : notebook.widgets.length;
      notebook.model.sharedModel.insertCell(cellIndex, {
        cell_type: 'code',
        source: codeSnippet
      });
      
      // Activate the new cell
      notebook.activeCellIndex = cellIndex;
    }

    // Show success notification
    this.showNotification(`✅ Inserted ${selectedFunc.displayName}`, '#4caf50');
    
    // Reset dropdown
    this.dropdown.value = '';
    this.populateDropdown();
  }

  private generateCodeSnippet(func: SHMFunction): string {
    const params = func.parameters;
    const hasRequiredParams = params.some(p => !p.optional);
    
    // Generate parameter string
    let paramStrings: string[] = [];
    
    params.forEach(param => {
      let paramStr = `    ${param.name}=`;
      
      if (param.default && param.default !== 'None') {
        // Use default value
        paramStr += param.default;
      } else if (param.type.includes('array') || param.type.includes('ndarray')) {
        paramStr += 'data';  // Placeholder for array data
      } else if (param.type.includes('int')) {
        paramStr += '1';
      } else if (param.type.includes('float')) {
        paramStr += '1.0';
      } else if (param.type.includes('str')) {
        paramStr += "'value'";
      } else {
        paramStr += 'None';
      }
      
      // Add comment with type info
      if (param.description) {
        paramStr += `,  # ${param.description}`;
      } else {
        paramStr += `,  # ${param.type}`;
      }
      
      // Mark optional parameters
      if (param.optional) {
        paramStr += ' (optional)';
      }
      
      paramStrings.push(paramStr);
    });
    
    // Generate function call
    let code = `# ${func.description}\n`;
    
    // Determine output variable name
    const outputVar = this.suggestOutputVariable(func.name);
    
    if (paramStrings.length > 0) {
      code += `${outputVar} = shmtools.${func.name}(\n${paramStrings.join('\n')}\n)`;
    } else {
      code += `${outputVar} = shmtools.${func.name}()`;
    }
    
    return code;
  }

  private suggestOutputVariable(funcName: string): string {
    // Suggest meaningful output variable names based on function
    const suggestions: { [key: string]: string } = {
      'psd_welch': 'frequencies, psd',
      'ar_model': 'ar_coeffs, rmse',
      'score_pca': 'scores',
      'learn_pca': 'pca_model',
      'score_mahalanobis': 'distances',
      'learn_mahalanobis': 'maha_model',
      'filter_butterworth': 'filtered_data',
      'statistical_moments': 'moments',
    };
    
    // Check if we have a specific suggestion
    for (const [pattern, suggestion] of Object.entries(suggestions)) {
      if (funcName.includes(pattern)) {
        return suggestion;
      }
    }
    
    // Generic output name
    return 'result';
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
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 3000);
  }
}

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

    console.log(`🔍 Target line ${targetLine}, column ${targetCol}`);
    
    // For multi-line function calls, we need to analyze the entire code block
    return this.findParameterInMultiLineFunction(code, cursorPos, lines, targetLine, targetCol);
  }

  private findParameterInMultiLineFunction(code: string, cursorPos: number, lines: string[], targetLine: number, targetCol: number): ParameterContext | null {
    // First, find the function call that contains our cursor position
    console.log('🔍 About to call extractFunctionCallAtPosition...');
    const functionCall = this.extractFunctionCallAtPosition(code, cursorPos);
    console.log('🔍 extractFunctionCallAtPosition returned:', functionCall);
    if (!functionCall) {
      console.log('🔍 No function call found at cursor position');
      return null;
    }

    console.log(`🔍 Found function call: ${functionCall.functionName}`);
    console.log(`🔍 Function call span: ${functionCall.startPos} - ${functionCall.endPos}`);

    // Parse parameters within the function call
    return this.findParameterInFunctionCall(functionCall, cursorPos);
  }

  private extractFunctionCallAtPosition(code: string, cursorPos: number): {
    functionName: string;
    startPos: number;
    endPos: number;
    fullText: string;
    parametersText: string;
  } | null {
    console.log(`🔍 *** STARTING extractFunctionCallAtPosition ***`);
    console.log(`🔍 Extracting function call at position ${cursorPos}`);
    console.log(`🔍 Code length: ${code.length}`);
    
    // Add a test to see if the method is working at all
    if (!code || code.length === 0) {
      console.log(`🔍 ❌ Code is empty or undefined`);
      return null;
    }
    
    if (cursorPos < 0 || cursorPos >= code.length) {
      console.log(`🔍 ❌ Cursor position ${cursorPos} is out of bounds for code length ${code.length}`);
      return null;
    }
    
    // Strategy: Look for function patterns in the code and check if cursor is within their scope
    const functionPattern = /(\w+(?:\.\w+)*)\s*\(/g;
    let match;
    let bestMatch = null;
    let matchCount = 0;
    
    // Find all function calls in the code
    console.log(`🔍 Searching for function patterns in code...`);
    console.log(`🔍 Using regex: ${functionPattern}`);
    console.log(`🔍 Code to search:`);
    console.log(code);
    while ((match = functionPattern.exec(code)) !== null) {
      matchCount++;
      const matchStart = match.index;
      const functionName = match[1];
      const openParenPos = match.index + match[0].length - 1;
      
      console.log(`🔍 Match #${matchCount}: Found function "${functionName}" at position ${matchStart}, opening paren at ${openParenPos}`);
      console.log(`🔍 Match details: "${match[0]}" (full match)`);
      console.log(`🔍 Function name captured: "${match[1]}"`);
      console.log(`🔍 Code at match start: "${code.substring(matchStart, matchStart + 20)}"`);
      console.log(`🔍 Character at open paren pos: "${code[openParenPos]}"`);
      console.log(`🔍 Context: "${code.substring(Math.max(0, matchStart-5), matchStart+15)}"`);
      console.log(`🔍 Cursor ${cursorPos} vs match range ${matchStart}-???`);
      
      // Find the matching closing parenthesis
      let parenCount = 0;
      let functionEnd = -1;
      
      for (let i = openParenPos; i < code.length; i++) {
        if (code[i] === '(') {
          parenCount++;
        } else if (code[i] === ')') {
          parenCount--;
          if (parenCount === 0) {
            functionEnd = i + 1;
            break;
          }
        }
      }
      
      if (functionEnd === -1) {
        console.log(`🔍 No closing paren found for "${functionName}"`);
        continue;
      }
      
      console.log(`🔍 Function "${functionName}" spans ${matchStart} to ${functionEnd}, cursor at ${cursorPos}`);
      console.log(`🔍 Checking if ${cursorPos} >= ${matchStart} && ${cursorPos} <= ${functionEnd}`);
      console.log(`🔍 First condition: ${cursorPos >= matchStart}, Second condition: ${cursorPos <= functionEnd}`);
      
      // Check if cursor is within this function call
      if (cursorPos >= matchStart && cursorPos <= functionEnd) {
        console.log(`🔍 ✅ Cursor is within function "${functionName}"!`);
        
        // If we have multiple nested functions, prefer the innermost one
        if (!bestMatch || (matchStart > bestMatch.startPos)) {
          const fullText = code.substring(matchStart, functionEnd);
          const parametersText = code.substring(openParenPos + 1, functionEnd - 1);
          
          // Extract just the function name part (after the last dot)
          const nameParts = functionName.split('.');
          const simpleName = nameParts[nameParts.length - 1];
          
          bestMatch = {
            functionName: simpleName,
            startPos: matchStart,
            endPos: functionEnd,
            fullText,
            parametersText
          };
        }
      }
    }
    
    console.log(`🔍 Total matches found: ${matchCount}`);
    
    if (bestMatch) {
      console.log(`🔍 ✅ Returning best match: ${bestMatch.functionName} (${bestMatch.startPos}-${bestMatch.endPos})`);
    } else {
      console.log(`🔍 ❌ No function call found containing cursor position ${cursorPos}`);
    }
    
    console.log(`🔍 *** ENDING extractFunctionCallAtPosition ***`);
    return bestMatch;
  }

  private findParameterInFunctionCall(functionCall: any, cursorPos: number): ParameterContext | null {
    const { functionName, startPos, parametersText } = functionCall;
    
    // Parse parameters from the parameters text (could be multi-line)
    const parameters = this.parseParameters(parametersText);
    
    console.log(`🔍 Parsed ${parameters.length} parameters:`, parameters);

    // Find which parameter the cursor is in
    // Calculate the position relative to the start of the parameters (after opening parenthesis)
    let parenPos = startPos + functionName.length;
    while (parenPos < functionCall.endPos && functionCall.fullText[parenPos - startPos] !== '(') {
      parenPos++;
    }
    const parametersStartPos = parenPos + 1;
    const relativePos = cursorPos - parametersStartPos;
    
    console.log(`🔍 Function "${functionName}" starts at ${startPos}, parameters start at ${parametersStartPos}`);
    console.log(`🔍 Cursor at absolute ${cursorPos}, relative to parameters: ${relativePos}`);
    console.log(`🔍 Parameters text: "${parametersText}"`);
    console.log(`🔍 Character at cursor in full code: "${functionCall.fullText[cursorPos - startPos] || 'END'}"`);
    
    for (const param of parameters) {
      const paramStart = param.startPos;
      const paramEnd = param.endPos;
      
      console.log(`🔍 Parameter "${param.name}"`);
      console.log(`   📍 Value: "${param.value}"`);
      console.log(`   📍 Range: [${paramStart}-${paramEnd}] (relative to parameters)`);
      console.log(`   📍 Absolute range: [${parametersStartPos + paramStart}-${parametersStartPos + paramEnd}]`);
      console.log(`   📍 Text at range: "${parametersText.substring(paramStart, paramEnd)}"`);
      console.log(`   📍 Is cursor in range? ${relativePos >= paramStart && relativePos <= paramEnd} (cursor=${relativePos})`);
      
      // Be more generous with the range check - include some margin
      const margin = 2; // Allow 2 characters margin
      if (relativePos >= (paramStart - margin) && relativePos <= (paramEnd + margin)) {
        console.log(`✅ Found parameter "${param.name}" at cursor position (with margin)`);
        
        // Calculate absolute positions for replacement
        const absoluteStart = parametersStartPos + paramStart;
        const absoluteEnd = parametersStartPos + paramEnd;
        
        console.log(`🔧 Replacement range: [${absoluteStart}-${absoluteEnd}]`);
        
        return {
          parameterName: param.name,
          currentValue: param.value,
          functionName: functionName,
          position: { line: 0, ch: 0 }, // Will be calculated properly in replacement
          replacementRange: { 
            start: absoluteStart,
            end: absoluteEnd
          }
        };
      }
    }
    
    console.log(`🔍 Cursor not in any parameter value using complex parsing`);
    
    // Fallback: try simpler approach
    console.log(`🔍 Trying fallback simple parameter detection...`);
    return this.fallbackParameterDetection(parametersText, relativePos, functionName, parametersStartPos);
  }

  private fallbackParameterDetection(parametersText: string, relativePos: number, functionName: string, parametersStartPos: number): ParameterContext | null {
    // Simple approach: split by commas and look for param=value patterns
    console.log(`🔍 Fallback: analyzing text around position ${relativePos}`);
    console.log(`🔍 Fallback: parameters text: "${parametersText}"`);
    
    // Get character at cursor position
    const charAtCursor = parametersText[relativePos] || '';
    console.log(`🔍 Fallback: character at cursor: "${charAtCursor}"`);
    
    // Find the parameter assignment that contains our position
    // Look backwards and forwards for = and comma/parenthesis
    let searchStart = relativePos;
    let searchEnd = relativePos;
    
    // Find the start of the current parameter assignment
    while (searchStart > 0 && parametersText[searchStart] !== ',' && parametersText[searchStart] !== '(') {
      searchStart--;
    }
    if (parametersText[searchStart] === ',' || parametersText[searchStart] === '(') {
      searchStart++; // Move past the comma or opening paren
    }
    
    // Find the end of the current parameter assignment
    while (searchEnd < parametersText.length - 1 && parametersText[searchEnd] !== ',' && parametersText[searchEnd] !== ')') {
      searchEnd++;
    }
    
    const paramText = parametersText.substring(searchStart, searchEnd).trim();
    console.log(`🔍 Fallback: found parameter text: "${paramText}"`);
    
    // Parse param=value from this text
    const match = paramText.match(/^\s*(\w+)\s*=\s*(.+?)\s*$/);
    if (match) {
      const paramName = match[1];
      const paramValue = match[2].trim();
      
      console.log(`🔍 Fallback: found parameter "${paramName}" = "${paramValue}"`);
      
      // Find the value position within the parameter text
      const equalPos = paramText.indexOf('=');
      let valueStart = equalPos + 1;
      while (valueStart < paramText.length && /\s/.test(paramText[valueStart])) {
        valueStart++;
      }
      const valueEnd = valueStart + paramValue.length;
      
      // Convert to absolute positions
      const absoluteValueStart = parametersStartPos + searchStart + valueStart;
      const absoluteValueEnd = parametersStartPos + searchStart + valueEnd;
      
      console.log(`🔍 Fallback: value positions [${absoluteValueStart}-${absoluteValueEnd}]`);
      
      return {
        parameterName: paramName,
        currentValue: paramValue,
        functionName: functionName,
        position: { line: 0, ch: 0 },
        replacementRange: { 
          start: absoluteValueStart,
          end: absoluteValueEnd
        }
      };
    }
    
    console.log(`🔍 Fallback: no parameter found`);
    return null;
  }

  private parseParameters(parametersText: string): Array<{
    name: string;
    value: string;
    startPos: number;
    endPos: number;
  }> {
    const parameters = [];
    
    // Remove comments and normalize whitespace while tracking positions
    let cleanText = '';
    let positionMap = []; // Maps clean position to original position
    
    for (let i = 0; i < parametersText.length; i++) {
      const char = parametersText[i];
      
      // Skip comments (# to end of line)
      if (char === '#') {
        let j = i;
        while (j < parametersText.length && parametersText[j] !== '\n') {
          j++;
        }
        if (j < parametersText.length && parametersText[j] === '\n') {
          cleanText += ' '; // Replace comment with single space
          positionMap.push(i);
        }
        i = j - 1; // Will be incremented by for loop
        continue;
      }
      
      cleanText += char;
      positionMap.push(i);
    }

    // Now parse parameters from clean text
    // Fixed pattern: parameter_name = value (handles multi-line with proper lookahead)
    // The key fix: allow any whitespace (including newlines) between comma and next parameter
    const paramRegex = /(\w+)\s*=\s*([^,)]+?)(?=\s*,\s*|\s*$|\s*\))/g;
    let match;
    
    while ((match = paramRegex.exec(cleanText)) !== null) {
      const paramName = match[1];
      let paramValue = match[2].trim();
      const matchStart = match.index;
      
      // Find the actual start of the value (after =)
      let valueSearchStart = matchStart + paramName.length;
      while (valueSearchStart < cleanText.length && cleanText[valueSearchStart] !== '=') {
        valueSearchStart++;
      }
      valueSearchStart++; // Skip the '=' character
      
      // Skip whitespace after =
      while (valueSearchStart < cleanText.length && /\s/.test(cleanText[valueSearchStart])) {
        valueSearchStart++;
      }
      
      // Find the end of the value (before comma, closing paren, or next parameter)
      let valueEnd = valueSearchStart;
      let parenDepth = 0;
      let inString = false;
      let stringChar = '';
      
      for (let i = valueSearchStart; i < cleanText.length; i++) {
        const char = cleanText[i];
        
        if (!inString) {
          if (char === '"' || char === "'") {
            inString = true;
            stringChar = char;
          } else if (char === '(') {
            parenDepth++;
          } else if (char === ')') {
            if (parenDepth === 0) {
              // End of function call
              break;
            }
            parenDepth--;
          } else if (char === ',' && parenDepth === 0) {
            // Check if this is the end of our parameter value
            // Look ahead to see if next non-whitespace is parameter_name=
            let lookahead = i + 1;
            while (lookahead < cleanText.length && /\s/.test(cleanText[lookahead])) {
              lookahead++;
            }
            if (lookahead < cleanText.length && /\w+\s*=/.test(cleanText.substring(lookahead))) {
              // This comma separates our parameter from the next one
              break;
            }
          }
        } else {
          if (char === stringChar && (i === 0 || cleanText[i-1] !== '\\')) {
            inString = false;
          }
        }
        
        valueEnd = i + 1;
      }
      
      // Extract the actual value text
      paramValue = cleanText.substring(valueSearchStart, valueEnd).trim();
      
      // Map back to original positions in the original text (with comments)
      const originalValueStart = positionMap[valueSearchStart] || valueSearchStart;
      const originalValueEnd = positionMap[Math.min(valueEnd - 1, positionMap.length - 1)] || (valueEnd - 1);
      
      console.log(`🔍 Parsed parameter "${paramName}" = "${paramValue}"`);
      console.log(`   📍 Clean text value start: ${valueSearchStart}, end: ${valueEnd}`);
      console.log(`   📍 Original positions: [${originalValueStart}-${originalValueEnd}]`);
      console.log(`   📍 Clean text at range: "${cleanText.substring(valueSearchStart, valueEnd)}"`);
      
      parameters.push({
        name: paramName,
        value: paramValue,
        startPos: originalValueStart,
        endPos: originalValueEnd + 1
      });
    }
    
    return parameters;
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
    
    // Use the replacement range from parameter context for precise replacement
    if (parameterContext.replacementRange) {
      const startPos = parameterContext.replacementRange.start;
      const endPos = parameterContext.replacementRange.end;
      
      console.log(`🔧 Replacing text from position ${startPos} to ${endPos}`);
      console.log(`🔧 Original value: "${currentText.substring(startPos, endPos)}"`);
      console.log(`🔧 New value: "${variable.name}"`);
      
      // Replace the exact range with the variable name
      const newText = currentText.substring(0, startPos) + 
                     variable.name + 
                     currentText.substring(endPos);
      
      // Update the cell content
      editor.model.sharedModel.setSource(newText);
    } else {
      // Fallback to old line-based replacement for compatibility
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