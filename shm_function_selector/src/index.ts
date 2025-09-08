import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { INotebookTracker } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { IConsoleTracker } from '@jupyterlab/console';
import { CodeConsole } from '@jupyterlab/console';
import { requestAPI } from './serverAPI';
import { MessageLoop } from '@lumino/messaging';
import { Widget } from '@lumino/widgets';

/**
 * The plugin registration information.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'shm-function-selector:plugin',
  description: 'SHM Function Selector for JupyterLab with function dropdown and context menu parameter linking',
  autoStart: true,
  requires: [INotebookTracker, IConsoleTracker],
  activate: activate
};

/**
 * Activate the SHM Function Selector extension.
 */
function activate(
  app: JupyterFrontEnd,
  notebookTracker: INotebookTracker,
  consoleTracker: IConsoleTracker
): void {
  console.log('🚀 SHM Function Selector JupyterLab extension activated!');

  // Initialize managers
  const contextMenuManager = new SHMContextMenuManager();
  const functionSelector = new SHMFunctionSelector(app, notebookTracker);
  
  // Set refresh callback on context menu manager
  contextMenuManager.setRefreshCallback(() => {
    functionSelector.forceNotebookRefresh();
  });

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

  // Set up keyboard shortcuts
  setupKeyboardShortcuts(app, notebookTracker, functionSelector);

  // Set up notebook tracking with full context menu functionality
  notebookTracker.widgetAdded.connect((sender, nbPanel) => {
    console.log('📓 Notebook added, setting up SHM context menu functionality');
    
    const notebook = nbPanel.content;
    
    // Note: Removed the red SHM Parameter Linker button per user request
    
    // REMOVED: Debugging click listener was interfering with JupyterLab's event handling
    
    // DISABLED: MutationObserver might be blocking rendering
    // Monitor DOM changes to see if cells are being added
    /*
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          mutation.addedNodes.forEach((node) => {
            if ((node as HTMLElement).classList?.contains('jp-Cell')) {
              console.log('🟢 CELL ADDED to DOM:', {
                cellType: (node as HTMLElement).classList.toString(),
                timestamp: Date.now()
              });
            }
          });
        }
      });
    });
    
    // Observe the notebook content area for cell additions
    const cellsContainer = notebook.node.querySelector('.jp-Notebook-cell') || notebook.node;
    observer.observe(cellsContainer.parentElement || cellsContainer, { 
      childList: true, 
      subtree: true 
    });
    */
    
    // Listen for right-click events ONLY on code cell editors to avoid interfering with JupyterLab
    // Using event delegation to handle dynamically added cells
    notebook.node.addEventListener('contextmenu', (event: MouseEvent) => {
      // Only process if the right-click is within a code cell editor area
      const target = event.target as HTMLElement;
      const codeEditor = target.closest('.jp-CodeCell .jp-Editor');
      
      if (!codeEditor) {
        // Not in a code cell editor, let JupyterLab handle it normally
        return;
      }
      
      console.log('🟡 CONTEXTMENU in code cell editor', {
        timestamp: Date.now()
      });
      
      const activeCell = notebook.activeCell;
      if (!activeCell || activeCell.model.type !== 'code') {
        return;
      }

      console.log('🚀 Right-click detected, altKey:', event.altKey, 'ctrlKey:', event.ctrlKey, 'shiftKey:', event.shiftKey);

      // Check if Alt/Option key is held for plotting mode
      if (event.altKey) {
        // Alt+Right-click: Show plotting menu for variables in the cell
        const cellCode = activeCell.editor?.model?.sharedModel?.getSource() || '';
        console.log('📝 Cell code for plotting:', cellCode);
        const allVariables = contextMenuManager.getAllVariablesFromCodeForPlotting(cellCode);
        console.log('🔍 Variables detected for plotting:', allVariables);
        
        if (allVariables.length > 0) {
          console.log('🎯 Plotting mode: Found variables:', allVariables);
          event.preventDefault();
          event.stopPropagation();
          
          if (allVariables.length > 1) {
            // Show menu with all variables from the assignment
            contextMenuManager.showMultiVariablePlottingMenu(event, allVariables, consoleTracker);
          } else {
            // Show single variable plotting menu
            contextMenuManager.showPlottingContextMenu(event, allVariables[0], consoleTracker);
          }
          return;
        } else {
          // No variables found for plotting
          const notification = document.createElement('div');
          notification.textContent = '📊 No output variables found in this cell';
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
          return;
        }
      }
      
      // Also check if we're right-clicking directly on an output area
      const outputVariable = contextMenuManager.detectOutputVariable(event, activeCell);
      if (outputVariable) {
        console.log('🎯 Output variable detected for plotting:', outputVariable);
        
        // Get all variables from the most recent assignment
        const cellCode = activeCell.editor?.model?.sharedModel?.getSource() || '';
        const allVariables = contextMenuManager.getAllVariablesFromCodeForPlotting(cellCode);
        
        event.preventDefault();
        event.stopPropagation();
        
        if (allVariables.length > 1) {
          // Show menu with all variables from the assignment
          contextMenuManager.showMultiVariablePlottingMenu(event, allVariables, consoleTracker);
        } else {
          // Show single variable plotting menu
          contextMenuManager.showPlottingContextMenu(event, outputVariable, consoleTracker);
        }
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
      const currentCellIndex = notebook.activeCellIndex;
      console.log(`🔍 Current cell index for context menu: ${currentCellIndex}`);
      
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
        
        // Show SHM context menu with current cell index and validation setting
        const enableValidation = functionSelector.getSettingValue('enableParameterValidation', false);
        contextMenuManager.showContextMenu(event, parameterContext, notebook, currentCellIndex, enableValidation);
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
      
      // Don't call notebook.update() here - it interferes with JupyterLab's 
      // internal state management and causes toolbar buttons to require 
      // double-clicks to work properly
    });
  });

  console.log('✅ SHM Function Selector setup complete - Function Dropdown (Phase 1) and Context Menu (Phase 3) loaded!');
}

export default plugin;

// ============================================================================
// KEYBOARD SHORTCUTS IMPLEMENTATION
// ============================================================================

function setupKeyboardShortcuts(
  app: JupyterFrontEnd, 
  notebookTracker: INotebookTracker, 
  functionSelector: SHMFunctionSelector
): void {
  console.log('⌨️ Setting up SHM keyboard shortcuts');

  // Shortcut 1: Ctrl+Shift+F - Open function browser
  app.commands.addCommand('shm-selector:open-function-browser', {
    label: 'Open SHM Function Browser',
    caption: 'Open the SHM function browser',
    execute: () => {
      const activeNotebook = notebookTracker.currentWidget;
      if (activeNotebook) {
        const jfuseButton = activeNotebook.node.querySelector('.shm-jfuse-button') as HTMLElement;
        if (jfuseButton) {
          jfuseButton.click();
          console.log('📚 Opened function browser via keyboard shortcut');
        } else {
          console.log('⚠️ jFUSE button not found');
        }
      }
    }
  });

  // Shortcut 2: Ctrl+Shift+H - Show help for current function
  app.commands.addCommand('shm-selector:show-function-help', {
    label: 'Show SHM Function Help',
    caption: 'Show documentation for the current function under cursor',
    execute: () => {
      const activeNotebook = notebookTracker.currentWidget;
      if (activeNotebook) {
        const activeCell = activeNotebook.content.activeCell;
        if (activeCell && activeCell.model.type === 'code') {
          const editor = activeCell.editor;
          if (editor) {
            const cursor = editor.getCursorPosition();
            const code = editor.model.sharedModel.getSource();
            
            // Find function name at cursor position
            const functionName = extractFunctionNameAtCursor(code, cursor);
            if (functionName) {
              // Get the function from the selector and show its documentation
              const func = functionSelector.getFunctionByName(functionName);
              if (func) {
                functionSelector.showDocumentationPopup(func);
                console.log(`📖 Showed help for function: ${functionName}`);
              } else {
                showKeyboardNotification(`Function "${functionName}" not found in SHM library`, '#ff9800');
              }
            } else {
              showKeyboardNotification('No SHM function found at cursor position', '#ff9800');
            }
          }
        }
      }
    }
  });

  // Shortcut 5: Ctrl+Shift+S - Search functions
  app.commands.addCommand('shm-selector:search-functions', {
    label: 'Search SHM Functions',
    caption: 'Open function search dialog',
    execute: () => {
      showFunctionSearchDialog(functionSelector, notebookTracker);
    }
  });

  // Register keyboard bindings
  app.commands.addKeyBinding({
    command: 'shm-selector:open-function-browser',
    keys: ['Ctrl Shift F'],
    selector: '.jp-Notebook'
  });

  app.commands.addKeyBinding({
    command: 'shm-selector:show-function-help',
    keys: ['Ctrl Shift H'],
    selector: '.jp-Notebook'
  });

  app.commands.addKeyBinding({
    command: 'shm-selector:search-functions',
    keys: ['Ctrl Shift /'],
    selector: 'body'
  });

  console.log('✅ SHM keyboard shortcuts registered:');
  console.log('   📚 Ctrl+Shift+F - Open function browser');
  console.log('   📖 Ctrl+Shift+H - Show function help');
  console.log('   🔍 Ctrl+Shift+/ - Search functions');
}

// Helper functions for keyboard shortcuts

function extractFunctionNameAtCursor(code: string, cursor: any): string | null {
  const lines = code.split('\n');
  const line = lines[cursor.line] || '';
  
  // Look for function calls like shmtools.function_name or just function_name
  const beforeCursor = line.substring(0, cursor.column);
  const afterCursor = line.substring(cursor.column);
  
  // Pattern to match function names
  const functionPattern = /(?:shmtools\.)?(\w+)(?:_shm)?\s*\(/;
  
  // Look backwards from cursor for function call
  for (let i = beforeCursor.length; i >= 0; i--) {
    const segment = beforeCursor.substring(i) + afterCursor.substring(0, 20);
    const match = segment.match(functionPattern);
    if (match) {
      return match[1] + '_shm'; // Always add _shm suffix for internal lookup
    }
  }
  
  return null;
}

function showKeyboardNotification(message: string, color: string): void {
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

function showPopularFunctionsQuickSelect(
  functionSelector: SHMFunctionSelector, 
  notebookTracker: INotebookTracker
): void {
  // Create quick select overlay
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.35);
    z-index: 10000;
    display: flex;
    justify-content: center;
    align-items: center;
  `;

  const popup = document.createElement('div');
  popup.style.cssText = `
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    max-width: 400px;
    width: 90%;
  `;

  const title = document.createElement('h3');
  title.textContent = 'Popular SHM Functions';
  title.style.cssText = `
    margin: 0 0 16px 0;
    color: #333;
    font-size: 16px;
    text-align: center;
  `;

  // Popular functions list
  const popularFunctions = [
    'psd_welch_shm',
    'ar_model_shm',
    'score_pca_shm',
    'learn_pca_shm',
    'score_mahalanobis_shm',
    'learn_mahalanobis_shm',
    'filter_butterworth_shm',
    'statistical_moments_shm'
  ];

  const functionsList = document.createElement('div');
  
  popularFunctions.forEach((funcName, index) => {
    const func = functionSelector.getFunctionByName(funcName);
    if (func) {
      const item = document.createElement('div');
      item.style.cssText = `
        padding: 8px 12px;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: background 0.2s;
      `;

      const numberSpan = document.createElement('span');
      numberSpan.textContent = `${index + 1}. `;
      numberSpan.style.cssText = `
        font-weight: bold;
        color: #666;
        margin-right: 8px;
      `;

      const nameSpan = document.createElement('span');
      nameSpan.textContent = func.displayName;
      nameSpan.style.cssText = `
        font-weight: bold;
        color: #333;
      `;

      item.appendChild(numberSpan);
      item.appendChild(nameSpan);

      item.addEventListener('mouseenter', () => {
        item.style.background = '#f0f0f0';
      });

      item.addEventListener('mouseleave', () => {
        item.style.background = 'white';
      });

      item.addEventListener('click', () => {
        functionSelector.insertFunction(func);
        overlay.remove();
        showKeyboardNotification(`✅ Inserted ${func.displayName}`, '#4caf50');
      });

      functionsList.appendChild(item);
    }
  });

  const instructions = document.createElement('div');
  instructions.textContent = 'Click a function or press 1-8 to insert';
  instructions.style.cssText = `
    text-align: center;
    color: #666;
    font-size: 11px;
    margin-top: 12px;
  `;

  popup.appendChild(title);
  popup.appendChild(functionsList);
  popup.appendChild(instructions);
  overlay.appendChild(popup);

  // Add number key handlers
  const keyHandler = (e: KeyboardEvent) => {
    const num = parseInt(e.key);
    if (num >= 1 && num <= popularFunctions.length) {
      const funcName = popularFunctions[num - 1];
      const func = functionSelector.getFunctionByName(funcName);
      if (func) {
        functionSelector.insertFunction(func);
        overlay.remove();
        showKeyboardNotification(`✅ Inserted ${func.displayName}`, '#4caf50');
      }
    } else if (e.key === 'Escape') {
      overlay.remove();
    }
    document.removeEventListener('keydown', keyHandler);
  };

  // Close on overlay click
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      overlay.remove();
      document.removeEventListener('keydown', keyHandler);
    }
  });

  document.addEventListener('keydown', keyHandler);
  document.body.appendChild(overlay);
}

function smartParameterLink(notebook: any): void {
  const activeCell = notebook.activeCell;
  if (!activeCell || activeCell.model.type !== 'code') {
    showKeyboardNotification('No active code cell', '#ff9800');
    return;
  }

  const editor = activeCell.editor;
  if (!editor) return;

  const cursor = editor.getCursorPosition();
  const code = editor.model.sharedModel.getSource();
  
  // Find current line and check for parameter pattern
  const lines = code.split('\n');
  const currentLine = lines[cursor.line] || '';
  
  // Look for parameter=value pattern at cursor
  const paramMatch = currentLine.match(/(\w+)\s*=\s*([^,)]+)/g);
  if (paramMatch) {
    // Find the closest parameter to cursor position
    let targetParam = null;
    let minDistance = Infinity;
    
    paramMatch.forEach(match => {
      const paramIndex = currentLine.indexOf(match);
      const distance = Math.abs(paramIndex - cursor.column);
      if (distance < minDistance) {
        minDistance = distance;
        targetParam = match;
      }
    });
    
    if (targetParam) {
      const [paramName] = targetParam.split('=').map(s => s.trim());
      showKeyboardNotification(`🔗 Smart linking for parameter: ${paramName}`, '#2196f3');
      
      // Trigger context menu programmatically
      const contextMenuManager = new SHMContextMenuManager();
      const parameterContext = {
        parameterName: paramName,
        currentValue: 'None',
        functionName: 'unknown',
        position: cursor,
        replacementRange: { start: 0, end: 0 }
      };
      
      // Show context menu at a calculated position
      const fakeEvent = {
        pageX: window.innerWidth / 2,
        pageY: window.innerHeight / 2,
        preventDefault: () => {},
        stopPropagation: () => {}
      } as MouseEvent;
      
      contextMenuManager.showContextMenu(fakeEvent, parameterContext, notebook, notebook.activeCellIndex, false);
    } else {
      showKeyboardNotification('No parameter found at cursor position', '#ff9800');
    }
  } else {
    showKeyboardNotification('Cursor not on a parameter assignment', '#ff9800');
  }
}

function showFunctionSearchDialog(
  functionSelector: SHMFunctionSelector, 
  notebookTracker: INotebookTracker
): void {
  // Create search overlay
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.35);
    z-index: 10000;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 10vh;
  `;

  const popup = document.createElement('div');
  popup.style.cssText = `
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    max-width: 500px;
    width: 90%;
    max-height: 70vh;
    overflow-y: auto;
  `;

  const title = document.createElement('h3');
  title.textContent = 'Search SHM Functions';
  title.style.cssText = `
    margin: 0 0 16px 0;
    color: #333;
    font-size: 16px;
    text-align: center;
  `;

  const searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.placeholder = 'Type to search functions...';
  searchInput.style.cssText = `
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    margin-bottom: 16px;
    box-sizing: border-box;
  `;

  const resultsContainer = document.createElement('div');
  resultsContainer.style.cssText = `
    max-height: 300px;
    overflow-y: auto;
  `;

  popup.appendChild(title);
  popup.appendChild(searchInput);
  popup.appendChild(resultsContainer);
  overlay.appendChild(popup);

  // Keyboard navigation state
  let selectedIndex = -1;
  let searchResults: HTMLElement[] = [];

  // Search functionality
  let searchTimeout: number;
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      const query = searchInput.value.toLowerCase();
      searchResults = updateSearchResults(resultsContainer, functionSelector, query, cleanupAndClose);
      selectedIndex = -1; // Reset selection when search results change
    }, 200);
  });

  // Function to cleanup and close dialog
  const cleanupAndClose = () => {
    document.removeEventListener('keydown', keyboardHandler);
    if (overlay.parentNode) {
      overlay.remove();
    }
  };

  // Keyboard navigation handler
  const keyboardHandler = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      cleanupAndClose();
      return;
    }

    // Only handle arrow keys and Enter if we have search results
    if (searchResults.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      e.stopPropagation();
      selectedIndex = Math.min(selectedIndex + 1, searchResults.length - 1);
      updateSelectionHighlight(searchResults, selectedIndex);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      e.stopPropagation();
      selectedIndex = Math.max(selectedIndex - 1, -1);
      updateSelectionHighlight(searchResults, selectedIndex);
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      e.stopPropagation();
      const func = (searchResults[selectedIndex] as any).__functionData;
      if (func) {
        // Insert FIRST (like dropdown does)
        functionSelector.insertFunction(func);
        showKeyboardNotification(`✅ Inserted ${func.displayName}`, '#4caf50');
      }
      // Then clean up after (like dropdown does with closeDropdown)
      cleanupAndClose();
    }
  };

  // Close handlers
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      cleanupAndClose();
    }
  });

  document.addEventListener('keydown', keyboardHandler);

  document.body.appendChild(overlay);
  searchInput.focus();
}

function updateSearchResults(
  container: HTMLElement, 
  functionSelector: SHMFunctionSelector, 
  query: string, 
  cleanupCallback: () => void
): HTMLElement[] {
  container.innerHTML = '';
  const resultElements: HTMLElement[] = [];

  if (query.length < 2) {
    const placeholder = document.createElement('div');
    placeholder.textContent = 'Type at least 2 characters to search...';
    placeholder.style.cssText = `
      color: #666;
      font-style: italic;
      text-align: center;
      padding: 20px;
    `;
    container.appendChild(placeholder);
    return resultElements;
  }

  const functions = functionSelector.getAllFunctions();
  const filteredFunctions = functions.filter(func => 
    func.displayName.toLowerCase().includes(query) ||
    func.description.toLowerCase().includes(query) ||
    func.category.toLowerCase().includes(query) ||
    func.name.toLowerCase().includes(query)
  );

  if (filteredFunctions.length === 0) {
    const noResults = document.createElement('div');
    noResults.textContent = 'No functions found matching your search.';
    noResults.style.cssText = `
      color: #666;
      font-style: italic;
      text-align: center;
      padding: 20px;
    `;
    container.appendChild(noResults);
    return resultElements;
  }

  filteredFunctions.slice(0, 10).forEach(func => { // Limit to 10 results
    const item = document.createElement('div');
    item.className = 'search-result-item';
    item.style.cssText = `
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: 4px;
      margin-bottom: 8px;
      cursor: pointer;
      transition: background 0.2s;
    `;

    const nameDiv = document.createElement('div');
    nameDiv.textContent = func.displayName;
    nameDiv.style.cssText = `
      font-weight: bold;
      color: #333;
      margin-bottom: 4px;
    `;

    const descDiv = document.createElement('div');
    descDiv.textContent = func.description;
    descDiv.style.cssText = `
      color: #666;
      font-size: 12px;
      margin-bottom: 4px;
    `;

    const categoryDiv = document.createElement('div');
    categoryDiv.textContent = func.category;
    categoryDiv.style.cssText = `
      color: #999;
      font-size: 10px;
    `;

    item.appendChild(nameDiv);
    item.appendChild(descDiv);
    item.appendChild(categoryDiv);

    // Store function data on the element for keyboard navigation
    (item as any).__functionData = func;

    item.addEventListener('mouseenter', () => {
      // Clear keyboard selection when mouse is used
      updateSelectionHighlight(resultElements, -1);
      item.style.background = '#f0f0f0';
    });

    item.addEventListener('mouseleave', () => {
      item.style.background = 'white';
    });

    item.addEventListener('click', () => {
      // Insert FIRST (like dropdown does)
      functionSelector.insertFunction(func);
      showKeyboardNotification(`✅ Inserted ${func.displayName}`, '#4caf50');
      // Then clean up after (like dropdown does with closeDropdown)
      cleanupCallback();
    });

    container.appendChild(item);
    resultElements.push(item);
  });

  if (filteredFunctions.length > 10) {
    const moreResults = document.createElement('div');
    moreResults.textContent = `... and ${filteredFunctions.length - 10} more results`;
    moreResults.style.cssText = `
      color: #666;
      font-style: italic;
      text-align: center;
      padding: 12px;
    `;
    container.appendChild(moreResults);
  }

  return resultElements;
}

function updateSelectionHighlight(resultElements: HTMLElement[], selectedIndex: number): void {
  // Clear all highlights first
  resultElements.forEach((element, index) => {
    if (index === selectedIndex) {
      // Highlight the selected item
      element.style.background = '#cce7ff';
      element.style.color = 'black';
      // Ensure the selected item is visible
      element.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } else {
      // Reset non-selected items
      element.style.background = 'white';
      element.style.color = '';
    }
  });
}

// ============================================================================
// PHASE 1: FUNCTION SELECTOR DROPDOWN IMPLEMENTATION
// ============================================================================

interface SHMFunction {
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

interface CategoryNode {
  name: string;
  children: Map<string, CategoryNode>;
  functions: SHMFunction[];
  level: number;
}


class SHMFunctionSelector {
  private app: JupyterFrontEnd;
  private notebookTracker: INotebookTracker;
  private functions: SHMFunction[] = [];
  private moduleImports: string[] = [];  // Store imports here
  private dropdown: HTMLSelectElement | null = null;
  private recentlyUsed: string[] = [];
  
  // Keyboard navigation state
  private keyboardNavigationItems: HTMLElement[] = [];
  private selectedNavigationIndex: number = -1;
  private dropdownKeyboardHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor(app: JupyterFrontEnd, notebookTracker: INotebookTracker) {
    this.app = app;
    this.notebookTracker = notebookTracker;
    
    // Load functions from server
    this.loadFunctions();
    
    // Set up notebook tracking
    this.setupNotebookTracking();
  }

  /**
   * Force JupyterLab to update all notebooks to fix rendering issues
   * Call this after any event handlers that might block updates
   */
  public forceNotebookRefresh(): void {
    requestAnimationFrame(() => {
      // Get current notebook
      const currentWidget = this.notebookTracker.currentWidget;
      if (currentWidget && currentWidget.content) {
        const notebook = currentWidget.content;
        // Schedule a repaint/layout pass
        notebook.update();
        // Post explicit Lumino messages
        MessageLoop.postMessage(notebook, Widget.Msg.UpdateRequest);
        MessageLoop.postMessage(notebook, Widget.Msg.FitRequest);
        // Also update the panel
        currentWidget.update();
      }
      // Also try to update the shell
      if (this.app.shell) {
        MessageLoop.postMessage(this.app.shell as any, Widget.Msg.FitRequest);
      }
    });
  }

  private async loadModuleImports(): Promise<void> {
    try {
      console.log('📥 Loading module imports from server...');
      const response = await requestAPI<any>('imports');
      
      let imports: string[];
      if (typeof response === 'string') {
        imports = JSON.parse(response);
      } else if (Array.isArray(response)) {
        imports = response;
      } else {
        console.warn('Unexpected imports response type:', typeof response);
        imports = [];
      }
      
      this.moduleImports = imports;
      console.log(`✅ Loaded ${imports.length} module imports:`, imports);
    } catch (error) {
      console.error('❌ Failed to load module imports:', error);
      this.moduleImports = [];  // Fallback to empty
    }
  }

  private async loadFunctions(): Promise<void> {
    try {
      console.log('📥 Loading SHM functions from server...');
      console.log('📡 Making API request to: shm-function-selector/functions');
      const response = await requestAPI<any>('functions');
      
      // Check if response is a string that needs parsing
      let functions: SHMFunction[];
      if (typeof response === 'string') {
        console.log('📝 Response is string, parsing JSON...');
        functions = JSON.parse(response);
      } else if (Array.isArray(response)) {
        functions = response;
      } else {
        throw new Error(`Unexpected response type: ${typeof response}`);
      }
      
      this.functions = functions;
      console.log(`✅ Loaded ${functions.length} SHM functions`, functions.slice(0, 3));
      
      // Also load module imports
      await this.loadModuleImports();
      
      // Add special import function at the beginning of the list
      const importFunction: SHMFunction = {
        name: '__import_all_modules__',
        displayName: '📦 Import All Modules',
        category: '⚡ Quick Actions',
        module: 'builtin',
        signature: 'import_all_modules()',
        description: 'Add import statements for all available top-level modules',
        docstring: 'Imports all available modules like shmtools, examples, ladpackage',
        parameters: [],
        guiMetadata: {},
        returns: []
      };
      this.functions.unshift(importFunction);
      
      // If dropdown exists, populate it
      if (this.dropdown) {
        this.populateDropdown();
      }
      
    } catch (error) {
      console.error('❌ Failed to load SHM functions:', error);
      console.error('Error details:', error);
      console.error('Error type:', typeof error);
      console.error('Error message:', error?.message);
      console.error('Error stack:', error?.stack);
      
      // Show error notification
      this.showNotification('⚠️ Failed to load SHM functions. Check browser console.', '#ff9800');
    }
  }


  private setupNotebookTracking(): void {
    this.notebookTracker.widgetAdded.connect((sender, nbPanel) => {
      console.log('📓 Adding function selector to notebook toolbar');
      
      // Create the container for toolbar items
      const container = document.createElement('div');
      container.className = 'jp-Toolbar-item';
      container.style.cssText = `
        display: inline-flex;
        align-items: center;
        margin: 2px 5px;
        gap: 5px;
        flex-shrink: 0;
        white-space: nowrap;
        z-index: 1000;
      `;

      // Create compact jFUSE button
      const jfuseButton = document.createElement('button');
      jfuseButton.className = 'shm-jfuse-button';
      jfuseButton.textContent = 'jFUSE';
      jfuseButton.title = 'SHM Function Selector';
      jfuseButton.style.cssText = `
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 500;
        border: 1px solid #ccc;
        border-radius: 3px;
        background: white;
        cursor: pointer;
        min-width: fit-content;
        color: #333;
        transition: background-color 0.2s;
      `;

      // Add hover effect
      jfuseButton.addEventListener('mouseenter', () => {
        jfuseButton.style.backgroundColor = '#f0f0f0';
      });
      jfuseButton.addEventListener('mouseleave', () => {
        jfuseButton.style.backgroundColor = 'white';
      });

      // Add click handler to show the full menu overlay
      jfuseButton.addEventListener('click', (e) => {
        e.stopPropagation();
        this.showFunctionSelectorOverlay(nbPanel);
      });

      // Store reference for later use
      this.dropdown = document.createElement('select'); // Keep for compatibility
      this.dropdown.style.display = 'none';

      // Create settings button
      const settingsButton = document.createElement('button');
      settingsButton.textContent = '⚙️';
      settingsButton.title = 'SHM Extension Settings';
      settingsButton.style.cssText = `
        padding: 4px 6px;
        font-size: 11px;
        border: 1px solid #ccc;
        border-radius: 3px;
        background: white;
        cursor: pointer;
        min-width: 28px;
      `;

      settingsButton.addEventListener('click', () => {
        this.showSettingsPanel();
      });

      // Create help button
      const helpButton = document.createElement('button');
      helpButton.textContent = '❓';
      helpButton.title = 'SHM Extension Help';
      helpButton.style.cssText = `
        padding: 4px 6px;
        font-size: 11px;
        border: 1px solid #ccc;
        border-radius: 3px;
        background: white;
        cursor: pointer;
        min-width: 28px;
      `;

      helpButton.addEventListener('click', () => {
        this.showHelpPanel();
      });

      // Add elements to container
      container.appendChild(jfuseButton);
      container.appendChild(settingsButton);
      container.appendChild(helpButton);

      // Add to notebook toolbar
      const toolbar = nbPanel.toolbar;
      if (toolbar) {
        toolbar.node.appendChild(container);
        console.log('✅ Added compact jFUSE button to toolbar');
        
        // Load functions (but don't create dropdown)
        // Functions are loaded when overlay is shown
      }
    });
  }

  private populateDropdown(): void {
    if (!this.dropdown) return;

    // Replace simple dropdown with enhanced folding interface
    this.createFoldingDropdown();
  }

  private createFoldingDropdown(): void {
    if (!this.dropdown) return;

    // Clear existing content
    this.dropdown.innerHTML = '';
    this.dropdown.style.display = 'none';

    // Create the enhanced dropdown container
    const container = this.dropdown.parentElement!;
    let enhancedDropdown = container.querySelector('.shm-enhanced-dropdown') as HTMLElement;
    
    if (!enhancedDropdown) {
      enhancedDropdown = document.createElement('div');
      enhancedDropdown.className = 'shm-enhanced-dropdown';
      enhancedDropdown.style.cssText = `
        position: relative;
        min-width: 400px;
        max-width: 400px;
        width: 400px;
      `;
      container.appendChild(enhancedDropdown);
    }

    enhancedDropdown.innerHTML = '';

    // Create the trigger button
    const triggerButton = document.createElement('button');
    triggerButton.className = 'shm-dropdown-trigger';
    triggerButton.textContent = 'jFUSE';
    triggerButton.style.cssText = `
      width: 100%;
      padding: 6px 12px;
      font-size: 11px;
      border: 1px solid #ccc;
      border-radius: 3px;
      background: white;
      cursor: pointer;
      text-align: left;
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;

    // Add dropdown arrow
    const arrow = document.createElement('span');
    arrow.textContent = '▼';
    arrow.style.cssText = `
      font-size: 8px;
      transition: transform 0.2s;
    `;
    triggerButton.appendChild(arrow);

    // Create the dropdown content
    const dropdownContent = document.createElement('div');
    dropdownContent.className = 'shm-dropdown-content';
    dropdownContent.style.cssText = `
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: white;
      border: 1px solid #ccc;
      border-top: none;
      border-radius: 0 0 4px 4px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      max-height: min(400px, 60vh);
      overflow-y: auto;
      z-index: 1000;
      display: none;
    `;
    
    // Add responsive behavior for mobile
    const addResponsiveStyles = () => {
      if (window.innerWidth < 768) {
        dropdownContent.style.position = 'fixed';
        dropdownContent.style.top = '50%';
        dropdownContent.style.left = '50%';
        dropdownContent.style.transform = 'translate(-50%, -50%)';
        dropdownContent.style.right = 'auto';
        dropdownContent.style.width = '90vw';
        dropdownContent.style.maxWidth = '400px';
        dropdownContent.style.maxHeight = '70vh';
        dropdownContent.style.borderRadius = '8px';
        dropdownContent.style.border = '1px solid #ccc';
      } else {
        dropdownContent.style.position = 'absolute';
        dropdownContent.style.top = '100%';
        dropdownContent.style.left = '0';
        dropdownContent.style.transform = 'none';
        dropdownContent.style.right = '0';
        dropdownContent.style.width = 'auto';
        dropdownContent.style.maxHeight = 'min(400px, 60vh)';
        dropdownContent.style.borderRadius = '0 0 4px 4px';
        dropdownContent.style.borderTop = 'none';
      }
    };
    
    addResponsiveStyles();
    window.addEventListener('resize', addResponsiveStyles);

    this.populateFoldingContent(dropdownContent);

    // Add click handler for trigger
    triggerButton.addEventListener('click', (e) => {
      e.stopPropagation();
      const isVisible = dropdownContent.style.display !== 'none';
      
      if (isVisible) {
        dropdownContent.style.display = 'none';
        arrow.style.transform = 'rotate(0deg)';
        this.cleanupDropdownKeyboardNavigation();
      } else {
        dropdownContent.style.display = 'block';
        arrow.style.transform = 'rotate(180deg)';
        // Collapse all categories when dropdown reopens
        this.collapseAllCategories(dropdownContent);
        // Setup outside click handler when dropdown opens
        setupOutsideClickHandler();
        // Auto-focus the search box when dropdown opens
        setTimeout(() => {
          const searchBox = dropdownContent.querySelector('input') as HTMLInputElement;
          if (searchBox) {
            searchBox.focus();
          }
          this.setupDropdownKeyboardNavigation(dropdownContent);
          // Ensure navigation items are updated after categories are collapsed
          this.updateNavigableItems(dropdownContent);
        }, 50);
      }
    });

    // Store click handler so we can remove it later
    let outsideClickHandler: ((e: MouseEvent) => void) | null = null;
    
    // Only add outside click handler when dropdown is open
    const setupOutsideClickHandler = () => {
      if (!outsideClickHandler) {
        outsideClickHandler = (e: MouseEvent) => {
          if (!enhancedDropdown.contains(e.target as Node)) {
            dropdownContent.style.display = 'none';
            arrow.style.transform = 'rotate(0deg)';
            this.cleanupDropdownKeyboardNavigation();
            // Remove the handler when dropdown closes
            if (outsideClickHandler) {
              document.removeEventListener('click', outsideClickHandler);
              outsideClickHandler = null;
            }
            // Force refresh after handling click
            this.forceNotebookRefresh();
          } else {
            // Force refresh even when click is inside dropdown
            this.forceNotebookRefresh();
          }
        };
        // Use setTimeout to avoid catching the same click that opened it
        setTimeout(() => {
          if (outsideClickHandler && dropdownContent.style.display !== 'none') {
            document.addEventListener('click', outsideClickHandler);
          }
        }, 0);
      }
    };

    enhancedDropdown.appendChild(triggerButton);
    enhancedDropdown.appendChild(dropdownContent);
  }


  private populateFoldingContent(container: HTMLElement): void {
    container.innerHTML = '';

    // Add search box
    const searchBox = document.createElement('input');
    searchBox.type = 'text';
    searchBox.placeholder = '🔍 Search functions...';
    searchBox.style.cssText = `
      width: calc(100% - 16px);
      padding: 8px;
      margin: 8px;
      border: 1px solid #ddd;
      border-radius: 3px;
      font-size: 11px;
    `;
    container.appendChild(searchBox);

    // Add the main content
    this.populateFoldingContentWithoutSearch(container);

    // Add search functionality
    searchBox.addEventListener('input', (e) => {
      const searchTerm = (e.target as HTMLInputElement).value.toLowerCase();
      this.filterFunctions(container, searchTerm);
    });
  }

  private populateFoldingContentWithoutSearch(container: HTMLElement): void {
    // Add recently used section if any
    if (this.recentlyUsed.length > 0) {
      const recentSection = this.createFoldingSection('⏱️ Recently Used', true);
      
      this.recentlyUsed.forEach(funcName => {
        const func = this.functions.find(f => f.name === funcName);
        if (func) {
          const item = this.createFunctionItem(func, true);
          recentSection.content.appendChild(item);
        }
      });
      
      container.appendChild(recentSection.container);
    }

    // Group functions by nested category structure using "-" delimiter
    const categoryTree = this.buildCategoryTree(this.functions);
    this.renderCategoryTree(categoryTree, container);
  }

  private buildCategoryTree(functions: SHMFunction[]): CategoryNode {
    const root: CategoryNode = { 
      name: 'root', 
      children: new Map(), 
      functions: [],
      level: 0
    };

    functions.forEach(func => {
      const categoryParts = func.category.split(' - ').map(part => part.trim());
      let currentNode = root;

      // Navigate/create the tree structure
      categoryParts.forEach((part, index) => {
        if (!currentNode.children.has(part)) {
          currentNode.children.set(part, {
            name: part,
            children: new Map(),
            functions: [],
            level: index + 1
          });
        }
        currentNode = currentNode.children.get(part)!;
      });

      // Add function to the deepest level
      currentNode.functions.push(func);
    });

    return root;
  }

  private renderCategoryTree(node: CategoryNode, container: HTMLElement, parentExpanded: boolean = true): void {
    // Sort children by name
    const sortedChildren = Array.from(node.children.entries()).sort(([a], [b]) => a.localeCompare(b));
    
    sortedChildren.forEach(([categoryName, childNode]) => {
      const section = this.createFoldingSection(categoryName, false, childNode.level);
      
      // Render child categories recursively
      this.renderCategoryTree(childNode, section.content, false);
      
      // Add functions at this level
      if (childNode.functions.length > 0) {
        const sortedFuncs = childNode.functions.sort((a, b) => a.displayName.localeCompare(b.displayName));
        sortedFuncs.forEach(func => {
          const item = this.createFunctionItem(func, false);
          section.content.appendChild(item);
        });
      }
      
      container.appendChild(section.container);
    });
  }

  private createFoldingSection(title: string, expanded: boolean = false, level: number = 0): {
    container: HTMLElement;
    header: HTMLElement;
    content: HTMLElement;
  } {
    const container = document.createElement('div');
    container.className = 'shm-category-section';

    const header = document.createElement('div');
    header.className = 'shm-category-header';
    
    // Calculate indentation based on level
    const indent = level * 20; // 20px per level
    
    header.style.cssText = `
      padding: 8px 12px;
      padding-left: ${12 + indent}px;
      background: #f8f9fa;
      border-bottom: 1px solid #e9ecef;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: bold;
      font-size: 11px;
      color: #495057;
      user-select: none;
    `;

    const titleSpan = document.createElement('span');
    titleSpan.textContent = title;

    const arrow = document.createElement('span');
    arrow.textContent = expanded ? '▼' : '▶';
    arrow.style.cssText = `
      font-size: 8px;
      transition: transform 0.2s;
    `;

    header.appendChild(titleSpan);
    header.appendChild(arrow);

    const content = document.createElement('div');
    content.className = 'shm-category-content';
    content.style.cssText = `
      display: ${expanded ? 'block' : 'none'};
      border-bottom: 1px solid #e9ecef;
      padding-left: ${indent}px;
    `;

    // Add click handler for folding
    header.addEventListener('click', () => {
      const isExpanded = content.style.display !== 'none';
      
      if (isExpanded) {
        content.style.display = 'none';
        arrow.textContent = '▶';
      } else {
        content.style.display = 'block';
        arrow.textContent = '▼';
      }
    });

    container.appendChild(header);
    container.appendChild(content);

    return { container, header, content };
  }


  private collapseAllCategories(container: HTMLElement): void {
    // Find all category sections and collapse them
    const categorySections = container.querySelectorAll('.shm-category-section');
    categorySections.forEach(section => {
      const content = section.querySelector('.shm-category-content') as HTMLElement;
      const arrow = section.querySelector('.shm-category-header span:last-child') as HTMLElement;
      
      if (content && arrow) {
        content.style.display = 'none';
        arrow.textContent = '▶';
      }
    });
  }

  private createFunctionItem(func: SHMFunction, isRecent: boolean = false): HTMLElement {
    const item = document.createElement('div');
    item.className = 'shm-function-item';
    item.setAttribute('data-function-name', func.name); // Add data attribute for click handler
    item.style.cssText = `
      padding: 8px 16px;
      cursor: pointer;
      border-bottom: 1px solid #f1f3f4;
      transition: background 0.2s;
      display: flex;
      justify-content: space-between;
      align-items: center;
      ${isRecent ? 'background: #fff3e0;' : ''}
    `;

    // Create main content area
    const contentDiv = document.createElement('div');
    contentDiv.style.cssText = `
      flex: 1;
      min-width: 0;
    `;

    const nameDiv = document.createElement('div');
    nameDiv.style.cssText = `
      font-weight: bold;
      font-size: 11px;
      color: ${isRecent ? '#f57c00' : '#333'};
      margin-bottom: 2px;
    `;
    nameDiv.textContent = func.displayName;

    const descDiv = document.createElement('div');
    descDiv.style.cssText = `
      font-size: 9px;
      color: #666;
      line-height: 1.3;
    `;
    descDiv.textContent = func.description.substring(0, 60) + (func.description.length > 60 ? '...' : '');

    contentDiv.appendChild(nameDiv);
    contentDiv.appendChild(descDiv);

    // Create actions area with help button
    const actionsDiv = document.createElement('div');
    actionsDiv.style.cssText = `
      display: flex;
      gap: 4px;
      margin-left: 8px;
    `;

    const helpButton = document.createElement('button');
    helpButton.textContent = '📖';
    helpButton.title = 'Show function documentation';
    helpButton.style.cssText = `
      border: none;
      background: none;
      cursor: pointer;
      font-size: 10px;
      padding: 2px 4px;
      border-radius: 2px;
      opacity: 0.6;
      transition: opacity 0.2s, background 0.2s;
    `;

    // Add help button functionality
    helpButton.addEventListener('click', (e) => {
      e.stopPropagation();
      this.showFunctionDocumentation(func, helpButton);
    });

    helpButton.addEventListener('mouseenter', () => {
      helpButton.style.opacity = '1';
      helpButton.style.background = '#e3f2fd';
    });

    helpButton.addEventListener('mouseleave', () => {
      helpButton.style.opacity = '0.6';
      helpButton.style.background = 'none';
    });

    actionsDiv.appendChild(helpButton);

    item.appendChild(contentDiv);
    item.appendChild(actionsDiv);

    // Add hover effects
    item.addEventListener('mouseenter', () => {
      // Clear keyboard selection when mouse is used
      this.selectedNavigationIndex = -1;
      this.updateNavigationHighlight();
      item.style.background = isRecent ? '#fff8e1' : '#f8f9fa';
    });

    item.addEventListener('mouseleave', () => {
      item.style.background = isRecent ? '#fff3e0' : 'white';
    });

    // Add click handler for main area
    contentDiv.addEventListener('click', () => {
      this.selectFunction(func);
      this.closeDropdown();
      
      // Also close overlay if it exists
      const overlay = document.querySelector('.shm-function-overlay');
      if (overlay) {
        overlay.remove();
        this.cleanupDropdownKeyboardNavigation();
      }
    });

    return item;
  }

  private showFunctionDocumentation(func: SHMFunction, triggerElement: HTMLElement): void {
    // Remove existing popup if any
    const existingPopup = document.querySelector('.shm-documentation-popup');
    if (existingPopup) {
      existingPopup.remove();
    }

    // Check if we're inside the function selector overlay
    const isInFunctionSelector = !!document.querySelector('.shm-function-overlay');

    // Create documentation popup
    const popup = document.createElement('div');
    popup.className = 'shm-documentation-popup';
    popup.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: white;
      border: 1px solid #ccc;
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
      max-width: min(600px, 90vw);
      width: 90vw;
      max-height: 80vh;
      overflow-y: auto;
      z-index: ${isInFunctionSelector ? '10002' : '10001'};
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: clamp(10px, 2vw, 12px);
    `;
    
    // Add responsive font sizes for mobile
    if (window.innerWidth < 768) {
      popup.style.width = '95vw';
      popup.style.maxHeight = '85vh';
    }

    // Create popup content
    const content = this.createDocumentationContent(func);
    popup.appendChild(content);

    // Add close button
    const closeButton = document.createElement('button');
    closeButton.textContent = '✕';
    closeButton.style.cssText = `
      position: absolute;
      top: 8px;
      right: 8px;
      border: none;
      background: #f5f5f5;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      cursor: pointer;
      font-size: 12px;
      line-height: 1;
      color: #666;
    `;

    closeButton.addEventListener('click', () => {
      popup.remove();
      if (!isInFunctionSelector && overlay) {
        overlay.remove();
      }
    });

    closeButton.addEventListener('mouseenter', () => {
      closeButton.style.background = '#e0e0e0';
      closeButton.style.color = '#333';
    });

    closeButton.addEventListener('mouseleave', () => {
      closeButton.style.background = '#f5f5f5';
      closeButton.style.color = '#666';
    });

    popup.appendChild(closeButton);

    // Only add overlay if we're not already in the function selector
    let overlay: HTMLElement | null = null;
    if (!isInFunctionSelector) {
      overlay = document.createElement('div');
      overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.25);
        z-index: 10000;
      `;

      overlay.addEventListener('click', () => {
        popup.remove();
        overlay!.remove();
      });

      document.body.appendChild(overlay);
    }

    // Add to DOM
    document.body.appendChild(popup);

    // Focus management
    popup.focus();
    
    // Close on Escape key
    const escapeHandler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        popup.remove();
        if (!isInFunctionSelector && overlay) {
          overlay.remove();
        }
        document.removeEventListener('keydown', escapeHandler);
      }
    };
    document.addEventListener('keydown', escapeHandler);
  }

  private createDocumentationContent(func: SHMFunction): HTMLElement {
    const content = document.createElement('div');
    content.style.cssText = `
      padding: 20px;
      line-height: 1.5;
    `;

    // Header section (moved to top)
    const header = document.createElement('div');
    header.style.cssText = `
      border-bottom: 2px solid #e9ecef;
      padding-bottom: 16px;
      margin-bottom: 16px;
    `;

    const title = document.createElement('h2');
    title.textContent = func.displayName;
    title.style.cssText = `
      margin: 0 0 8px 0;
      color: #333;
      font-size: 18px;
      font-weight: bold;
    `;

    const subtitle = document.createElement('div');
    subtitle.textContent = `${func.name} • ${func.category}`;
    subtitle.style.cssText = `
      color: #666;
      font-size: 11px;
      margin-bottom: 8px;
    `;

    const description = document.createElement('div');
    description.textContent = func.description;
    description.style.cssText = `
      color: #555;
      font-size: 12px;
      font-style: italic;
    `;

    header.appendChild(title);
    header.appendChild(subtitle);
    header.appendChild(description);

    // Add verbose call if available
    if (func.guiMetadata && func.guiMetadata.verbose_call) {
      const verboseCallDiv = document.createElement('div');
      verboseCallDiv.style.cssText = `
        margin-top: 12px;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 10px;
      `;

      const verboseCallLabel = document.createElement('div');
      verboseCallLabel.textContent = 'Verbose Call Signature:';
      verboseCallLabel.style.cssText = `
        color: #666;
        font-size: 11px;
        font-weight: bold;
        margin-bottom: 4px;
      `;

      const verboseCallCode = document.createElement('div');
      verboseCallCode.textContent = func.guiMetadata.verbose_call;
      verboseCallCode.style.cssText = `
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 11px;
        color: #333;
      `;

      verboseCallDiv.appendChild(verboseCallLabel);
      verboseCallDiv.appendChild(verboseCallCode);
      header.appendChild(verboseCallDiv);
    }

    // Add header to content first
    content.appendChild(header);

    // Parameters section
    if (func.parameters && func.parameters.length > 0) {
      const parametersSection = this.createParametersSection(func.parameters, func);
      content.appendChild(parametersSection);
    }

    // Returns section
    if (func.returns && func.returns.length > 0) {
      const returnsSection = this.createReturnsSection(func.returns);
      content.appendChild(returnsSection);
    }

    // GUI metadata section (with Complexity removed)
    if (func.guiMetadata) {
      const metadataSection = this.createMetadataSection(func.guiMetadata);
      content.appendChild(metadataSection);
    }

    // Full docstring section
    if (func.docstring) {
      const docstringSection = this.createDocstringSection(func.docstring);
      content.appendChild(docstringSection);
    }

    return content;
  }

  private createParametersSection(parameters: any[], func?: SHMFunction): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      margin-bottom: 16px;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Parameters';
    title.style.cssText = `
      margin: 0 0 8px 0;
      color: #333;
      font-size: 14px;
      font-weight: bold;
    `;

    const paramList = document.createElement('div');

    parameters.forEach(param => {
      const paramItem = document.createElement('div');
      paramItem.style.cssText = `
        margin-bottom: 12px;
        padding: 8px;
        background: #f8f9fa;
        border-left: 3px solid ${param.optional ? '#ffc107' : '#28a745'};
        border-radius: 0 4px 4px 0;
      `;

      const paramHeader = document.createElement('div');
      paramHeader.style.cssText = `
        font-weight: bold;
        margin-bottom: 4px;
        color: #333;
      `;

      // Get friendly name from verbose_call if available
      const friendlyName = this.getParameterFriendlyName(param.name, func);
      const displayName = friendlyName !== param.name ? friendlyName : param.name;

      const paramName = document.createElement('span');
      paramName.textContent = displayName;
      paramName.style.cssText = `
        color: #0d47a1;
      `;

      // Show original parameter name if different from friendly name
      if (friendlyName !== param.name) {
        const originalName = document.createElement('span');
        originalName.textContent = ` (${param.name})`;
        originalName.style.cssText = `
          color: #999;
          font-size: 11px;
          font-weight: normal;
        `;
        paramName.appendChild(originalName);
      }

      const paramType = document.createElement('span');
      paramType.textContent = ` : ${param.type}`;
      paramType.style.cssText = `
        color: #666;
      `;

      const paramStatus = document.createElement('span');
      paramStatus.textContent = param.optional ? ' (optional)' : ' (required)';
      paramStatus.style.cssText = `
        color: ${param.optional ? '#f57c00' : '#2e7d2e'};
        font-size: 10px;
      `;

      paramHeader.appendChild(paramName);
      paramHeader.appendChild(paramType);
      paramHeader.appendChild(paramStatus);

      paramItem.appendChild(paramHeader);

      if (param.description) {
        const paramDesc = document.createElement('div');
        paramDesc.textContent = param.description;
        paramDesc.style.cssText = `
          color: #555;
          font-size: 11px;
          margin-bottom: 4px;
        `;
        paramItem.appendChild(paramDesc);
      }

      if (param.default) {
        const paramDefault = document.createElement('div');
        paramDefault.textContent = `Default: ${param.default}`;
        paramDefault.style.cssText = `
          color: #666;
          font-size: 10px;
          font-style: italic;
        `;
        paramItem.appendChild(paramDefault);
      }

      if (param.widget) {
        const widgetInfo = document.createElement('div');
        widgetInfo.textContent = `Widget: ${param.widget.widget || 'default'}`;
        widgetInfo.style.cssText = `
          color: #666;
          font-size: 10px;
          font-style: italic;
        `;
        paramItem.appendChild(widgetInfo);
      }

      paramList.appendChild(paramItem);
    });

    section.appendChild(title);
    section.appendChild(paramList);
    return section;
  }

  private createReturnsSection(returns: any[]): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      margin-bottom: 16px;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Returns';
    title.style.cssText = `
      margin: 0 0 8px 0;
      color: #333;
      font-size: 14px;
      font-weight: bold;
    `;

    const returnsList = document.createElement('div');

    returns.forEach(ret => {
      const returnItem = document.createElement('div');
      returnItem.style.cssText = `
        margin-bottom: 8px;
        padding: 8px;
        background: #e8f5e8;
        border-left: 3px solid #4caf50;
        border-radius: 0 4px 4px 0;
      `;

      const returnHeader = document.createElement('div');
      returnHeader.style.cssText = `
        font-weight: bold;
        margin-bottom: 4px;
        color: #333;
      `;

      const returnName = document.createElement('span');
      returnName.textContent = ret.name;
      returnName.style.cssText = `
        color: #2e7d2e;
      `;

      const returnType = document.createElement('span');
      returnType.textContent = ` : ${ret.type}`;
      returnType.style.cssText = `
        color: #666;
      `;

      returnHeader.appendChild(returnName);
      returnHeader.appendChild(returnType);

      returnItem.appendChild(returnHeader);

      if (ret.description) {
        const returnDesc = document.createElement('div');
        returnDesc.textContent = ret.description;
        returnDesc.style.cssText = `
          color: #555;
          font-size: 11px;
        `;
        returnItem.appendChild(returnDesc);
      }

      returnsList.appendChild(returnItem);
    });

    section.appendChild(title);
    section.appendChild(returnsList);
    return section;
  }

  private createMetadataSection(metadata: any): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      margin-bottom: 16px;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Additional Information';
    title.style.cssText = `
      margin: 0 0 8px 0;
      color: #333;
      font-size: 14px;
      font-weight: bold;
    `;

    const metadataGrid = document.createElement('div');
    metadataGrid.style.cssText = `
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 8px;
      background: #f0f4f8;
      padding: 12px;
      border-radius: 4px;
      border: 1px solid #e1e8ed;
    `;

    const metadataEntries = [
      ['Data Type', metadata.data_type],
      ['Output Type', metadata.output_type],
      ['MATLAB Equivalent', metadata.matlab_equivalent]
    ].filter(([_, value]) => value);

    metadataEntries.forEach(([key, value]) => {
      const keyElement = document.createElement('div');
      keyElement.textContent = `${key}:`;
      keyElement.style.cssText = `
        font-weight: bold;
        color: #333;
        font-size: 11px;
      `;

      const valueElement = document.createElement('div');
      valueElement.textContent = value;
      valueElement.style.cssText = `
        color: #555;
        font-size: 11px;
      `;

      metadataGrid.appendChild(keyElement);
      metadataGrid.appendChild(valueElement);
    });

    section.appendChild(title);
    section.appendChild(metadataGrid);
    return section;
  }

  private createDocstringSection(docstring: string): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      margin-bottom: 16px;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Full Documentation';
    title.style.cssText = `
      margin: 0 0 8px 0;
      color: #333;
      font-size: 14px;
      font-weight: bold;
    `;

    const docstringContent = document.createElement('div');
    docstringContent.textContent = docstring;
    docstringContent.style.cssText = `
      background: #fafafa;
      border: 1px solid #e0e0e0;
      border-radius: 4px;
      padding: 12px;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 10px;
      line-height: 1.4;
      color: #333;
      white-space: pre-wrap;
      max-height: 200px;
      overflow-y: auto;
    `;

    section.appendChild(title);
    section.appendChild(docstringContent);
    return section;
  }

  private filterFunctions(container: HTMLElement, searchTerm: string): void {
    const sections = container.querySelectorAll('.shm-category-section');
    
    // First pass: filter function items and mark which sections have matches
    const sectionsWithMatches = new Set<Element>();
    
    sections.forEach(section => {
      const items = section.querySelectorAll('.shm-function-item');
      let hasVisibleItems = false;

      items.forEach(item => {
        const nameElement = item.querySelector('div');
        const descElement = item.querySelector('div:last-child');
        const name = nameElement?.textContent?.toLowerCase() || '';
        const desc = descElement?.textContent?.toLowerCase() || '';

        const matches = name.includes(searchTerm) || desc.includes(searchTerm);
        (item as HTMLElement).style.display = matches ? 'flex' : 'none';

        if (matches) {
          hasVisibleItems = true;
          // Mark this section and all its parent sections as having matches
          let currentSection = section;
          while (currentSection) {
            sectionsWithMatches.add(currentSection);
            // Find parent section by going up the DOM tree
            const parentContent = currentSection.parentElement;
            if (parentContent && parentContent.classList.contains('shm-category-content')) {
              currentSection = parentContent.parentElement;
              if (currentSection && currentSection.classList.contains('shm-category-section')) {
                sectionsWithMatches.add(currentSection);
              } else {
                break;
              }
            } else {
              break;
            }
          }
        }
      });
    });
    
    // Second pass: show/hide sections and expand those with matches
    sections.forEach(section => {
      const hasMatches = sectionsWithMatches.has(section);
      const sectionEl = section as HTMLElement;
      sectionEl.style.display = hasMatches || searchTerm.length === 0 ? 'block' : 'none';

      // Expand or collapse the section content and arrow
      const content = section.querySelector('.shm-category-content') as HTMLElement;
      const arrow = section.querySelector('.shm-category-header span:last-child') as HTMLElement;
      if (content && arrow) {
        if (hasMatches && searchTerm.length > 0) {
          content.style.display = 'block';
          arrow.textContent = '▼';
        } else if (searchTerm.length === 0) {
          // Reset to default collapsed state when no search term
          content.style.display = 'none';
          arrow.textContent = '▶';
        } else {
          content.style.display = 'none';
          arrow.textContent = '▶';
        }
      }
    });
  }

  // Keyboard navigation methods
  private setupDropdownKeyboardNavigation(dropdownContent: HTMLElement): void {
    // Find all navigable function items
    this.updateNavigableItems(dropdownContent);
    this.selectedNavigationIndex = -1;

    // Create keyboard handler
    this.dropdownKeyboardHandler = (e: KeyboardEvent) => {
      // Only handle events when dropdown is actually visible
      if (dropdownContent.style.display === 'none') {
        return;
      }

      // Allow normal typing in search box
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT') {
        // Handle arrow keys and Enter in search box
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          this.selectedNavigationIndex = 0;
          this.updateNavigationHighlight();
          // Move focus away from search box to enable navigation
          (target as HTMLInputElement).blur();
        } else if (e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          this.closeDropdown();
        }
        return;
      }

      // Handle navigation keys when not in search box
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        this.selectedNavigationIndex = Math.min(
          this.selectedNavigationIndex + 1, 
          this.keyboardNavigationItems.length - 1
        );
        this.updateNavigationHighlight();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        this.selectedNavigationIndex = Math.max(this.selectedNavigationIndex - 1, -1);
        if (this.selectedNavigationIndex === -1) {
          // Return focus to search box
          const searchBox = dropdownContent.querySelector('input') as HTMLInputElement;
          if (searchBox) {
            searchBox.focus();
          }
        }
        this.updateNavigationHighlight();
      } else if (e.key === 'Enter' && this.selectedNavigationIndex >= 0) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        // Get the selected item
        const selectedItem = this.keyboardNavigationItems[this.selectedNavigationIndex];
        
        // Check if it's a category header or function item
        if (selectedItem.classList.contains('shm-category-header')) {
          // Toggle category expand/collapse
          selectedItem.click();
          // Update navigable items after toggling
          setTimeout(() => {
            this.updateNavigableItems(dropdownContent);
            // Keep selection on the same category header
            const newIndex = this.keyboardNavigationItems.indexOf(selectedItem);
            if (newIndex !== -1) {
              this.selectedNavigationIndex = newIndex;
            }
            this.updateNavigationHighlight();
          }, 50);
        } else if (selectedItem.classList.contains('shm-function-item')) {
          // Insert function as before
          const contentDiv = selectedItem.querySelector('div') as HTMLElement;
          if (contentDiv) {
            contentDiv.click();
          }
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        this.closeDropdown();
      } else {
        // For any other key, focus search box and let user type
        const searchBox = dropdownContent.querySelector('input') as HTMLInputElement;
        if (searchBox && e.key.length === 1) {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          searchBox.focus();
          // Add the typed character to search box
          searchBox.value += e.key;
          searchBox.dispatchEvent(new Event('input'));
        }
      }
      
      // Force refresh after handling keyboard events
      this.forceNotebookRefresh();
    };

    // Add event listener in bubble phase (false) to not interfere with JupyterLab
    document.addEventListener('keydown', this.dropdownKeyboardHandler, false);

    // Note: Navigation items are now updated directly in the main search input handler
    // to avoid timing conflicts between filtering and navigation updates
  }

  private updateNavigableItems(dropdownContent: HTMLElement): void {
    // Get all category headers and function items
    const allItems: HTMLElement[] = [];
    
    // Only get top-level category sections (direct children)
    const topLevelCategories = Array.from(dropdownContent.children).filter(child => 
      child.classList.contains('shm-category-section')
    );
    
    topLevelCategories.forEach(section => {
      const sectionElement = section as HTMLElement;
      
      // Skip hidden sections
      if (sectionElement.style.display === 'none') {
        return;
      }
      
      // Add the category header
      const header = section.querySelector('.shm-category-header') as HTMLElement;
      if (header) {
        allItems.push(header);
      }
      
      // Check if category is expanded
      const content = section.querySelector('.shm-category-content') as HTMLElement;
      if (content && content.style.display !== 'none') {
        // Recursively add subcategory headers and functions
        this.addNavigableItemsFromContent(content, allItems);
      }
    });
    
    console.log('[SHM] Updated navigable items:', allItems.length, 'items');
    console.log('[SHM] Categories only?', allItems.every(item => item.classList.contains('shm-category-header')));
    console.log('[SHM] First 5 items:', allItems.slice(0, 5).map(item => 
      item.classList.contains('shm-category-header') ? 
        'Category: ' + item.textContent?.trim() : 
        'Function: ' + item.textContent?.trim()
    ));
    
    this.keyboardNavigationItems = allItems;
  }

  private addNavigableItemsFromContent(content: HTMLElement, allItems: HTMLElement[]): void {
    // Get direct child elements that are category sections or function items
    Array.from(content.children).forEach(child => {
      if (child.classList.contains('shm-category-section')) {
        // It's a subcategory
        const subSection = child as HTMLElement;
        if (subSection.style.display !== 'none') {
          const header = subSection.querySelector('.shm-category-header') as HTMLElement;
          if (header) {
            allItems.push(header);
          }
          
          // Check if subcategory is expanded
          const subContent = subSection.querySelector('.shm-category-content') as HTMLElement;
          if (subContent && subContent.style.display !== 'none') {
            // Recursively add items from subcategory
            this.addNavigableItemsFromContent(subContent, allItems);
          }
        }
      } else if (child.classList.contains('shm-function-item')) {
        // It's a function item
        const funcElement = child as HTMLElement;
        if (funcElement.style.display !== 'none') {
          allItems.push(funcElement);
        }
      }
    });
  }

  private updateNavigationHighlight(): void {
    this.keyboardNavigationItems.forEach((item, index) => {
      if (index === this.selectedNavigationIndex) {
        item.style.background = '#cce7ff';
        item.style.color = 'black';
        item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        item.style.background = '';
        item.style.color = '';
      }
    });
  }

  private closeDropdown(): void {
    const enhancedDropdown = document.querySelector('.shm-enhanced-dropdown') as HTMLElement;
    if (enhancedDropdown) {
      const dropdownContent = enhancedDropdown.querySelector('.shm-dropdown-content') as HTMLElement;
      const arrow = enhancedDropdown.querySelector('.shm-dropdown-trigger span') as HTMLElement;
      
      if (dropdownContent) dropdownContent.style.display = 'none';
      if (arrow) arrow.style.transform = 'rotate(0deg)';
      
      this.cleanupDropdownKeyboardNavigation();
    }
  }

  private cleanupDropdownKeyboardNavigation(): void {
    if (this.dropdownKeyboardHandler) {
      document.removeEventListener('keydown', this.dropdownKeyboardHandler, true);
      this.dropdownKeyboardHandler = null;
    }
    this.keyboardNavigationItems = [];
    this.selectedNavigationIndex = -1;
  }

  private selectFunction(func: SHMFunction): void {
    // Add to recently used (max 5)
    this.recentlyUsed = [func.name, ...this.recentlyUsed.filter(n => n !== func.name)].slice(0, 5);
    
    // Insert the function directly
    this.insertFunctionDirect(func);
  }

  private insertFunctionDirect(func: SHMFunction): void {
    console.log('🔧 Inserting function directly:', func.name);
    
    // Generate code snippet
    const codeSnippet = this.generateCodeSnippet(func);
    console.log('📝 Generated code snippet:', codeSnippet);
    
    // Get current notebook widget
    const currentWidget = this.notebookTracker.currentWidget;
    if (!currentWidget) {
      console.log('❌ No current notebook widget');
      this.showNotification('No active notebook found', '#ff9800');
      return;
    }

    const notebook = currentWidget.content;
    
    // Insert new cells after current cell
    console.log('📄 Inserting function cells after current cell');
    
    const currentCellIndex = notebook.activeCellIndex;
    const markdownContent = `## ${func.displayName}\n\n${func.description}`;
    
    if (currentCellIndex >= 0 && currentCellIndex < notebook.widgets.length) {
      // Check if current cell has content
      const currentCellModel = notebook.model.cells.get(currentCellIndex);
      const cellHasContent = currentCellModel && currentCellModel.sharedModel.getSource().trim().length > 0;
      
      // Determine insertion index based on whether current cell has content
      let insertIndex = currentCellIndex;
      if (cellHasContent) {
        // If current cell has content, insert after it
        insertIndex = currentCellIndex + 1;
      } else {
        // If current cell is empty, delete it and use its position
        notebook.model.sharedModel.deleteCell(currentCellIndex);
        // After deletion, the insertion point is exactly where the deleted cell was
        insertIndex = currentCellIndex;
      }
      
      // Insert all cells at once using insertCells for better rendering
      notebook.model.sharedModel.insertCells(insertIndex, [
        {
          cell_type: 'markdown',
          source: markdownContent
        },
        {
          cell_type: 'code',
          source: codeSnippet
        },
        {
          cell_type: 'code',
          source: ''
        }
      ]);
      
      // Set active cell immediately (not in requestAnimationFrame)
      // This follows JupyterLab's own pattern
      notebook.activeCellIndex = insertIndex + 2;
      notebook.deselectAll();
      notebook.activate();
      
      // Focus the editor after a minimal delay for DOM update
      setTimeout(() => {
        const targetCell = notebook.widgets[insertIndex + 2];
        if (targetCell) {
          notebook.select(targetCell);
          if (targetCell.editor) {
            targetCell.editor.focus();
            console.log('✅ Focused on new empty cell');
          }
        }
      }, 0);
    } else {
      // No active cell, create cells at the end
      const insertIndex = notebook.widgets.length;
      // Insert all cells at once
      notebook.model.sharedModel.insertCells(insertIndex, [
        {
          cell_type: 'markdown',
          source: markdownContent
        },
        {
          cell_type: 'code',
          source: codeSnippet
        },
        {
          cell_type: 'code',
          source: ''
        }
      ]);
      
      // Set active cell immediately
      notebook.activeCellIndex = insertIndex + 2;
      notebook.deselectAll();
      notebook.activate();
      
      // Focus after minimal delay
      setTimeout(() => {
        const targetCell = notebook.widgets[insertIndex + 2];
        if (targetCell) {
          notebook.select(targetCell);
          if (targetCell.editor) {
            targetCell.editor.focus();
          }
        }
      }, 0);
    }
    
    console.log('✅ Successfully inserted function with markdown description and empty cell');

    // Show success notification
    this.showNotification(`✅ Inserted ${func.displayName}`, '#4caf50');
    
    // Update the hidden dropdown for compatibility
    if (this.dropdown) {
      this.dropdown.value = func.name;
    }
    
    // Force notebook refresh to ensure proper rendering
    this.forceNotebookRefresh();
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
    
    // Use current cell for markdown and insert code cell after
    const notebook = nbPanel.content;
    const currentCellIndex = notebook.activeCellIndex;
    const markdownContent = `## ${selectedFunc.displayName}\n\n${selectedFunc.description}`;
    
    if (currentCellIndex >= 0 && currentCellIndex < notebook.widgets.length) {
      // Delete current cell and insert markdown + code cells
      notebook.model.sharedModel.deleteCell(currentCellIndex);
      
      // After deletion, the insertion point is exactly where the deleted cell was
      const insertIndex = currentCellIndex;
      
      // Insert both cells at once
      notebook.model.sharedModel.insertCells(insertIndex, [
        {
          cell_type: 'markdown',
          source: markdownContent
        },
        {
          cell_type: 'code',
          source: codeSnippet
        }
      ]);
      
      // Set active cell immediately
      notebook.activeCellIndex = insertIndex + 1;
      notebook.deselectAll();
      notebook.activate();
      
      // Focus after minimal delay
      setTimeout(() => {
        const codeTargetCell = notebook.widgets[insertIndex + 1];
        if (codeTargetCell) {
          notebook.select(codeTargetCell);
          if (codeTargetCell.editor) {
            codeTargetCell.editor.focus();
            // Position cursor at first parameter (None value)
            const lines = codeSnippet.split('\n');
            for (let i = 0; i < lines.length; i++) {
              const paramIndex = lines[i].indexOf('=None');
              if (paramIndex !== -1) {
                codeTargetCell.editor.setCursorPosition({ line: i, column: paramIndex + 1 });
                break;
              }
            }
          }
        }
      }, 0);
    } else {
      // No active cell, create both cells at the end
      const insertIndex = notebook.widgets.length;
      // Insert both cells at once
      notebook.model.sharedModel.insertCells(insertIndex, [
        {
          cell_type: 'markdown',
          source: markdownContent
        },
        {
          cell_type: 'code',
          source: codeSnippet
        }
      ]);
      
      // Set active cell immediately
      notebook.activeCellIndex = insertIndex + 1;
      notebook.deselectAll();
      notebook.activate();
      
      // Focus after minimal delay
      setTimeout(() => {
        const targetCell = notebook.widgets[insertIndex + 1];
        if (targetCell) {
          notebook.select(targetCell);
          if (targetCell.editor) {
            targetCell.editor.focus();
          }
        }
      }, 0);
    }

    // Show success notification
    this.showNotification(`✅ Inserted ${selectedFunc.displayName}`, '#4caf50');
    
    // Reset dropdown
    this.dropdown.value = '';
    this.populateDropdown();
    
    // Force notebook refresh to ensure proper rendering
    this.forceNotebookRefresh();
  }

  private generateCodeSnippet(func: SHMFunction): string {
    // Special handling for import all modules
    if (func.name === '__import_all_modules__') {
      return this.moduleImports.length > 0 
        ? this.moduleImports.join('\n')
        : '# No modules available to import';
    }

    const params = func.parameters;
    const hasRequiredParams = params.some(p => !p.optional);
    
    // Generate parameter string with enhanced defaults and validation
    let paramStrings: string[] = [];
    
    params.forEach((param, index) => {
      let paramStr = `    ${param.name}=`;
      let paramValue = this.getEnhancedParameterDefault(param);
      
      paramStr += paramValue;
      
      // Add comprehensive comment with validation info
      let comment = this.generateParameterComment(param, func);
      
      // Only add comma if this is not the last parameter
      if (index < params.length - 1) {
        paramStr += `,  # ${comment}`;
      } else {
        paramStr += `  # ${comment}`;
      }
      
      paramStrings.push(paramStr);
    });
    
    // Generate function call with enhanced output handling
    let code = this.generateFunctionHeader(func);
    
    // Add verbose_call comment if available
    if (func.guiMetadata && func.guiMetadata.verbose_call) {
      code += `# ${func.guiMetadata.verbose_call}\n`;
    }
    
    // Determine output variables based on return info
    const outputVar = this.suggestOutputVariables(func);
    
    if (paramStrings.length > 0) {
      code += `${outputVar} = ${func.module}.${func.name}(\n${paramStrings.join('\n')}\n)`;
    } else {
      code += `${outputVar} = ${func.module}.${func.name}()`;
    }
    
    // Add validation comments if validation rules exist
    if (this.hasValidationRules(func)) {
      code += '\n\n# Validation: ' + this.generateValidationComment(func);
    }
    
    return code;
  }

  private getEnhancedParameterDefault(param: any): string {
    // Priority 1: Use GUI widget default if available
    if (param.widget && param.widget.default) {
      return this.ensureProperQuoting(param.widget.default, param.type);
    }
    
    // Priority 2: Use function signature default
    if (param.default && param.default !== '<inspect.Parameter.empty>') {
      // Handle None as a valid default value
      if (param.default === 'None') {
        return 'None';
      }
      return this.ensureProperQuoting(param.default, param.type);
    }
    
    // Priority 3: Smart defaults based on parameter name and type
    const paramName = param.name.toLowerCase();
    
    // Data parameters
    if (['data', 'x', 'y', 'input_data', 'features', 'signals'].includes(paramName)) {
      return 'None';
    }
    
    // Sampling frequency parameters
    if (['fs', 'sampling_rate', 'sample_rate', 'freq'].includes(paramName)) {
      return '1000.0';
    }
    
    // Order parameters
    if (['order', 'ar_order', 'n_components', 'n_features'].includes(paramName)) {
      return '10';
    }
    
    // Window parameters
    if (['window', 'window_type'].includes(paramName)) {
      return "'hann'";
    }
    
    // Segment length parameters
    if (['nperseg', 'n_per_seg', 'segment_length'].includes(paramName)) {
      return '256';
    }
    
    // File parameters - only provide default if no actual default exists
    if (['filename', 'filepath', 'path'].includes(paramName)) {
      // If the function already has None as default, respect that
      if (param.default === 'None' || param.default === null || param.default === undefined) {
        return 'None';
      }
      // Otherwise provide a reasonable example
      return "'data.csv'";
    }
    
    // Type-based defaults
    if (param.type.includes('array') || param.type.includes('ndarray')) {
      return 'None';
    } else if (param.type.includes('int')) {
      return '1';
    } else if (param.type.includes('float')) {
      return '1.0';
    } else if (param.type.includes('str')) {
      return "'value'";
    } else if (param.type.includes('bool')) {
      return 'True';
    } else {
      return 'None';
    }
  }

  private generateParameterComment(param: any, func?: SHMFunction): string {
    let comment = '';
    
    // Try to get human-friendly name from verbose_call first
    const friendlyName = this.getParameterFriendlyName(param.name, func);
    if (friendlyName && friendlyName !== param.name) {
      comment += friendlyName;
    } else if (param.description) {
      // Add description if available
      comment += param.description;
    } else {
      comment += param.type;
    }
    
    // Add units for frequency parameters
    const paramName = param.name.toLowerCase();
    if (['fs', 'sampling_rate', 'sample_rate', 'freq'].includes(paramName)) {
      comment += ' Hz';
    }
    
    // Add validation info
    if (param.validation && param.validation.length > 0) {
      const validationInfo = param.validation.map((rule: any) => {
        if (rule.type === 'range') {
          return `range: ${rule.min}-${rule.max}`;
        } else if (rule.type === 'choice') {
          return `options: ${rule.options.join(', ')}`;
        } else if (rule.type === 'file_format') {
          return `formats: ${rule.formats.join(', ')}`;
        }
        return '';
      }).filter(Boolean).join(', ');
      
      if (validationInfo) {
        comment += `, ${validationInfo}`;
      }
    }
    
    // Mark optional parameters
    if (param.optional) {
      comment += ' (optional)';
    } else {
      comment += ' (required)';
    }
    
    return comment;
  }

  private getParameterFriendlyName(paramName: string, func?: SHMFunction): string {
    if (!func || !func.guiMetadata || !func.guiMetadata.verbose_call) {
      return paramName;
    }

    const verboseCall = func.guiMetadata.verbose_call;
    
    // Parse verbose_call format: [outputs] = FunctionName (Input1, Input2, ...)
    // Extract the part after the equals and function name, within parentheses
    const match = verboseCall.match(/=\s*[^(]+\s*\(([^)]+)\)/);
    if (!match) {
      return paramName;
    }

    const parametersPart = match[1];
    const friendlyNames = parametersPart.split(',').map(name => name.trim());
    
    // Map parameter names to their positions in the function signature
    const paramNames = func.parameters.map(p => p.name);
    const paramIndex = paramNames.indexOf(paramName);
    
    if (paramIndex >= 0 && paramIndex < friendlyNames.length) {
      return friendlyNames[paramIndex];
    }
    
    return paramName;
  }

  private ensureProperQuoting(value: string, paramType: string): string {
    // If the value is already properly quoted, return as is
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      return value;
    }
    
    // Check if this should be a string based on type information
    const isStringType = paramType.toLowerCase().includes('str') || 
                        paramType.includes('<class \'str\'>') ||
                        paramType.includes('typing.Literal') ||
                        paramType.includes('str]'); // for Union[str, ...]
    
    // Also check if the value looks like it should be a string:
    // - Contains only letters (likely an enum/choice value)
    // - Is a common Python literal that should be quoted
    const looksLikeString = /^[a-zA-Z][a-zA-Z0-9_]*$/.test(value) && 
                           !['True', 'False', 'None'].includes(value) &&
                           isNaN(Number(value));
    
    if (isStringType || looksLikeString) {
      // Add quotes if it's clearly a string
      return `'${value}'`;
    }
    
    // For other types, return as is
    return value;
  }

  private generateFunctionHeader(func: SHMFunction): string {
    // No longer include header in code snippet since it goes in markdown cell
    return '';
  }

  private suggestOutputVariables(func: SHMFunction): string {
    // Use return info if available
    if (func.returns && func.returns.length > 0) {
      const returnNames = func.returns.map((ret: any) => ret.name).filter(Boolean);
      if (returnNames.length > 1) {
        // Clean up the return names and remove any invalid characters
        const cleanNames = returnNames.map(name => {
          // Remove any invalid characters like '.', keep only valid Python identifiers
          return name.replace(/[^a-zA-Z0-9_]/g, '').trim();
        }).filter(name => name.length > 0); // Remove empty names
        
        if (cleanNames.length > 1) {
          return cleanNames.join(', ');
        } else if (cleanNames.length === 1) {
          return cleanNames[0];
        }
      } else if (returnNames.length === 1) {
        const cleanName = returnNames[0].replace(/[^a-zA-Z0-9_]/g, '').trim();
        if (cleanName.length > 0) {
          return cleanName;
        }
      }
    }
    
    // Fall back to name-based suggestions
    return this.suggestOutputVariable(func.name);
  }

  private hasValidationRules(func: SHMFunction): boolean {
    return func.parameters.some((param: any) => param.validation && param.validation.length > 0);
  }

  private generateValidationComment(func: SHMFunction): string {
    const validationComments = func.parameters
      .filter((param: any) => param.validation && param.validation.length > 0)
      .map((param: any) => `${param.name}: ${param.validation.map((rule: any) => rule.type).join(', ')}`)
      .join('; ');
    
    return validationComments || 'Parameter validation available';
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

  // Helper methods for keyboard shortcuts integration
  public getFunctionByName(name: string): SHMFunction | null {
    return this.functions.find(f => f.name === name) || null;
  }

  public getAllFunctions(): SHMFunction[] {
    return this.functions;
  }

  public insertFunction(func: SHMFunction): void {
    // Use the new direct insertion method
    this.insertFunctionDirect(func);
  }

  public showDocumentationPopup(func: SHMFunction): void {
    this.showFunctionDocumentation(func, document.body);
  }

  private showFunctionSelectorOverlay(nbPanel: any): void {
    // Remove existing overlay if any
    const existingOverlay = document.querySelector('.shm-function-overlay');
    if (existingOverlay) {
      existingOverlay.remove();
    }

    // Create overlay
    const overlay = document.createElement('div');
    overlay.className = 'shm-function-overlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.25);
      z-index: 10000;
      display: flex;
      justify-content: center;
      align-items: center;
    `;

    // Create function selector panel
    const panel = document.createElement('div');
    panel.className = 'shm-function-selector-panel';
    panel.style.cssText = `
      background: white;
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.2);
      max-width: min(500px, 90vw);
      width: 90vw;
      max-height: 80vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    `;

    // Create header
    const header = document.createElement('div');
    header.style.cssText = `
      padding: 16px 20px;
      border-bottom: 1px solid #e0e0e0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #f8f9fa;
    `;

    const title = document.createElement('h3');
    title.textContent = 'jFUSE - SHM Function Selector';
    title.style.cssText = `
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      color: #333;
    `;

    // Create close button
    const closeButton = document.createElement('button');
    closeButton.textContent = '✕';
    closeButton.style.cssText = `
      border: none;
      background: transparent;
      font-size: 20px;
      cursor: pointer;
      color: #666;
      padding: 0;
      width: 24px;
      height: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
    `;

    closeButton.addEventListener('click', () => {
      overlay.remove();
      this.cleanupDropdownKeyboardNavigation();
    });

    header.appendChild(title);
    header.appendChild(closeButton);
    panel.appendChild(header);

    // Create content container
    const contentContainer = document.createElement('div');
    contentContainer.style.cssText = `
      flex: 1;
      overflow-y: auto;
      padding: 0;
    `;

    // Add search box
    const searchContainer = document.createElement('div');
    searchContainer.style.cssText = `
      padding: 12px 20px;
      background: white;
      border-bottom: 1px solid #e0e0e0;
      position: sticky;
      top: 0;
      z-index: 10;
    `;

    const searchBox = document.createElement('input');
    searchBox.type = 'text';
    searchBox.placeholder = '🔍 Search functions...';
    searchBox.style.cssText = `
      width: 100%;
      padding: 8px 12px;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-size: 13px;
      outline: none;
    `;

    searchBox.addEventListener('focus', () => {
      searchBox.style.borderColor = '#4CAF50';
    });

    searchBox.addEventListener('blur', () => {
      searchBox.style.borderColor = '#ddd';
    });

    searchContainer.appendChild(searchBox);
    contentContainer.appendChild(searchContainer);

    // Create functions list container
    const functionsContainer = document.createElement('div');
    functionsContainer.style.cssText = `
      padding: 8px 0;
    `;

    // Populate with functions (without adding another search box)
    this.populateFoldingContentWithoutSearch(functionsContainer);

    // Update search functionality
    searchBox.addEventListener('input', () => {
      const searchTerm = searchBox.value.toLowerCase();
      
      // Use the filterFunctions method which properly handles all filtering
      this.filterFunctions(functionsContainer, searchTerm);
      
      // Update keyboard navigation items after filtering
      this.updateNavigableItems(functionsContainer);
      this.selectedNavigationIndex = -1;
      this.updateNavigationHighlight();
    });

    // Note: Click handlers are already attached to individual function items
    // in createFunctionItem(), so we don't need another delegated handler here.
    // The items will handle their own clicks and close the dropdown.

    contentContainer.appendChild(functionsContainer);
    panel.appendChild(contentContainer);

    // Add panel to overlay
    overlay.appendChild(panel);

    // Close on overlay click
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.remove();
        this.cleanupDropdownKeyboardNavigation();
      }
    });

    // Close on Escape key
    const escapeHandler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        overlay.remove();
        this.cleanupDropdownKeyboardNavigation();
        document.removeEventListener('keydown', escapeHandler);
      }
    };
    document.addEventListener('keydown', escapeHandler);

    // Add to DOM
    document.body.appendChild(overlay);

    // Focus search box after a short delay
    setTimeout(() => {
      searchBox.focus();
      this.setupDropdownKeyboardNavigation(functionsContainer);
    }, 100);
  }

  private showSettingsPanel(): void {
    // Remove existing settings panel if any
    const existingPanel = document.querySelector('.shm-settings-panel');
    if (existingPanel) {
      existingPanel.remove();
    }

    // Create settings overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.25);
      z-index: 10000;
      display: flex;
      justify-content: center;
      align-items: center;
    `;

    // Create settings panel
    const panel = document.createElement('div');
    panel.className = 'shm-settings-panel';
    panel.style.cssText = `
      background: white;
      border-radius: 8px;
      padding: 24px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.2);
      max-width: min(500px, 90vw);
      width: 90vw;
      max-height: 80vh;
      overflow-y: auto;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 12px;
    `;

    // Create panel content
    const content = this.createSettingsContent();
    panel.appendChild(content);

    // Add close button
    const closeButton = document.createElement('button');
    closeButton.textContent = '✕';
    closeButton.style.cssText = `
      position: absolute;
      top: 12px;
      right: 12px;
      border: none;
      background: #f5f5f5;
      border-radius: 50%;
      width: 28px;
      height: 28px;
      cursor: pointer;
      font-size: 14px;
      line-height: 1;
      color: #666;
    `;

    closeButton.addEventListener('click', () => {
      overlay.remove();
    });

    panel.appendChild(closeButton);
    overlay.appendChild(panel);

    // Close on overlay click
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.remove();
      }
    });

    // Close on Escape key
    const escapeHandler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        overlay.remove();
        document.removeEventListener('keydown', escapeHandler);
      }
    };
    document.addEventListener('keydown', escapeHandler);

    document.body.appendChild(overlay);
  }

  private createSettingsContent(): HTMLElement {
    const content = document.createElement('div');

    // Title
    const title = document.createElement('h2');
    title.textContent = 'SHM Function Selector Settings';
    title.style.cssText = `
      margin: 0 0 20px 0;
      color: #333;
      font-size: 18px;
      text-align: center;
    `;

    // Settings sections
    const settingsForm = document.createElement('div');

    // Auto-insert setting
    const autoInsertSection = this.createSettingSection(
      'Auto Function Insertion',
      'Automatically insert function when selected from dropdown',
      'checkbox',
      'autoInsert',
      this.getSettingValue('autoInsert', true)
    );

    // Show recently used setting
    const recentlyUsedSection = this.createSettingSection(
      'Show Recently Used',
      'Display recently used functions at the top of the dropdown',
      'checkbox',
      'showRecentlyUsed',
      this.getSettingValue('showRecentlyUsed', true)
    );

    // Function count setting
    const functionCountSection = this.createSettingSection(
      'Recently Used Count',
      'Number of recently used functions to remember',
      'number',
      'recentlyUsedCount',
      this.getSettingValue('recentlyUsedCount', 5)
    );

    // Context menu delay setting
    const contextMenuSection = this.createSettingSection(
      'Context Menu Sensitivity',
      'Right-click sensitivity for parameter detection',
      'select',
      'contextMenuSensitivity',
      this.getSettingValue('contextMenuSensitivity', 'normal'),
      ['high', 'normal', 'low']
    );

    // Keyboard shortcuts enabled
    const keyboardSection = this.createSettingSection(
      'Enable Keyboard Shortcuts',
      'Enable Ctrl+Shift+[F,H,I,L,S] shortcuts',
      'checkbox',
      'keyboardShortcuts',
      this.getSettingValue('keyboardShortcuts', true)
    );

    // Function documentation mode
    const docModeSection = this.createSettingSection(
      'Documentation Mode',
      'How to display function documentation',
      'select',
      'documentationMode',
      this.getSettingValue('documentationMode', 'popup'),
      ['popup', 'inline', 'sidebar']
    );

    // Parameter validation setting
    const validationSection = this.createSettingSection(
      'Enable Parameter Validation',
      'Validate parameter types when linking variables (disabled by default for flexibility)',
      'checkbox',
      'enableParameterValidation',
      this.getSettingValue('enableParameterValidation', false)
    );

    settingsForm.appendChild(autoInsertSection);
    settingsForm.appendChild(recentlyUsedSection);
    settingsForm.appendChild(functionCountSection);
    settingsForm.appendChild(contextMenuSection);
    settingsForm.appendChild(keyboardSection);
    settingsForm.appendChild(docModeSection);
    settingsForm.appendChild(validationSection);

    // Action buttons
    const buttonSection = document.createElement('div');
    buttonSection.style.cssText = `
      display: flex;
      gap: 12px;
      justify-content: center;
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid #eee;
    `;

    const saveButton = document.createElement('button');
    saveButton.textContent = 'Save Settings';
    saveButton.style.cssText = `
      padding: 8px 16px;
      background: #4caf50;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    `;

    const resetButton = document.createElement('button');
    resetButton.textContent = 'Reset to Defaults';
    resetButton.style.cssText = `
      padding: 8px 16px;
      background: #ff9800;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    `;

    saveButton.addEventListener('click', () => {
      this.saveSettings(settingsForm);
      this.showNotification('Settings saved successfully', '#4caf50');
      document.querySelector('.shm-settings-panel')?.parentElement?.remove();
    });

    resetButton.addEventListener('click', () => {
      this.resetSettings();
      this.showNotification('Settings reset to defaults', '#ff9800');
      document.querySelector('.shm-settings-panel')?.parentElement?.remove();
    });

    buttonSection.appendChild(saveButton);
    buttonSection.appendChild(resetButton);

    content.appendChild(title);
    content.appendChild(settingsForm);
    content.appendChild(buttonSection);

    return content;
  }


  private createSettingSection(
    label: string,
    description: string,
    type: string,
    key: string,
    value: any,
    options?: string[]
  ): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      margin-bottom: 20px;
      padding: 16px;
      border: 1px solid #e0e0e0;
      border-radius: 4px;
      background: #fafafa;
    `;

    const labelElement = document.createElement('div');
    labelElement.textContent = label;
    labelElement.style.cssText = `
      font-weight: bold;
      margin-bottom: 4px;
      color: #333;
    `;

    const descElement = document.createElement('div');
    descElement.textContent = description;
    descElement.style.cssText = `
      font-size: 11px;
      color: #666;
      margin-bottom: 8px;
    `;

    let inputElement: HTMLElement;

    if (type === 'checkbox') {
      inputElement = document.createElement('input');
      (inputElement as HTMLInputElement).type = 'checkbox';
      (inputElement as HTMLInputElement).checked = value;
    } else if (type === 'number') {
      inputElement = document.createElement('input');
      (inputElement as HTMLInputElement).type = 'number';
      (inputElement as HTMLInputElement).value = value.toString();
      (inputElement as HTMLInputElement).min = '1';
      (inputElement as HTMLInputElement).max = '10';
    } else if (type === 'select' && options) {
      inputElement = document.createElement('select');
      options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option.charAt(0).toUpperCase() + option.slice(1);
        if (option === value) {
          optionElement.selected = true;
        }
        (inputElement as HTMLSelectElement).appendChild(optionElement);
      });
    } else {
      inputElement = document.createElement('input');
      (inputElement as HTMLInputElement).type = 'text';
      (inputElement as HTMLInputElement).value = value.toString();
    }

    inputElement.setAttribute('data-setting-key', key);
    inputElement.style.cssText = `
      padding: 4px 8px;
      border: 1px solid #ccc;
      border-radius: 3px;
      font-size: 11px;
    `;

    section.appendChild(labelElement);
    section.appendChild(descElement);
    section.appendChild(inputElement);

    return section;
  }

  public getSettingValue(key: string, defaultValue: any): any {
    try {
      const stored = localStorage.getItem(`shm-selector-${key}`);
      if (stored !== null) {
        return typeof defaultValue === 'boolean' ? stored === 'true' : 
               typeof defaultValue === 'number' ? parseInt(stored) : stored;
      }
    } catch (e) {
      console.warn(`Failed to get setting ${key}:`, e);
    }
    return defaultValue;
  }

  private saveSettings(form: HTMLElement): void {
    const inputs = form.querySelectorAll('[data-setting-key]');
    inputs.forEach(input => {
      const key = input.getAttribute('data-setting-key')!;
      let value: string;
      
      if (input.getAttribute('type') === 'checkbox') {
        value = (input as HTMLInputElement).checked.toString();
      } else {
        value = (input as HTMLInputElement | HTMLSelectElement).value;
      }
      
      try {
        localStorage.setItem(`shm-selector-${key}`, value);
      } catch (e) {
        console.warn(`Failed to save setting ${key}:`, e);
      }
    });

    // Apply settings immediately
    this.applySettings();
  }

  private resetSettings(): void {
    const keys = [
      'autoInsert',
      'showRecentlyUsed', 
      'recentlyUsedCount',
      'contextMenuSensitivity',
      'keyboardShortcuts',
      'documentationMode',
      'enableParameterValidation'
    ];

    keys.forEach(key => {
      try {
        localStorage.removeItem(`shm-selector-${key}`);
      } catch (e) {
        console.warn(`Failed to reset setting ${key}:`, e);
      }
    });

    this.applySettings();
  }

  private applySettings(): void {
    // Apply recently used count setting
    const maxRecentlyUsed = this.getSettingValue('recentlyUsedCount', 5);
    this.recentlyUsed = this.recentlyUsed.slice(0, maxRecentlyUsed);

    console.log('✅ SHM settings applied');
  }

  private showHelpPanel(): void {
    // Remove existing help panel if any
    const existingPanel = document.querySelector('.shm-help-panel');
    if (existingPanel) {
      existingPanel.remove();
    }

    // Create help overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.25);
      z-index: 10000;
      display: flex;
      justify-content: center;
      align-items: center;
    `;

    // Create help panel
    const panel = document.createElement('div');
    panel.className = 'shm-help-panel';
    panel.style.cssText = `
      background: white;
      border-radius: 8px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
      padding: 24px;
      max-width: 600px;
      max-height: 80vh;
      overflow-y: auto;
      position: relative;
    `;

    // Create panel content
    const content = this.createHelpContent();
    panel.appendChild(content);

    // Add close button
    const closeButton = document.createElement('button');
    closeButton.textContent = '✕';
    closeButton.style.cssText = `
      position: absolute;
      top: 12px;
      right: 12px;
      background: none;
      border: none;
      font-size: 18px;
      cursor: pointer;
      color: #666;
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
    `;

    closeButton.addEventListener('click', () => {
      overlay.remove();
    });

    closeButton.addEventListener('mouseenter', () => {
      closeButton.style.background = '#f5f5f5';
    });

    closeButton.addEventListener('mouseleave', () => {
      closeButton.style.background = 'none';
    });

    panel.appendChild(closeButton);

    // Close on overlay click
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.remove();
      }
    });

    // Close on Escape key
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        overlay.remove();
        document.removeEventListener('keydown', handleEscape);
      }
    };
    document.addEventListener('keydown', handleEscape);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);
  }

  private createHelpContent(): HTMLElement {
    const content = document.createElement('div');

    // Title
    const title = document.createElement('h2');
    title.textContent = '📖 SHM Function Selector - Help & Usage';
    title.style.cssText = `
      margin: 0 0 20px 0;
      color: #333;
      font-size: 20px;
      text-align: center;
      border-bottom: 2px solid #e3f2fd;
      padding-bottom: 12px;
    `;

    // Introduction
    const intro = document.createElement('p');
    intro.innerHTML = `
      The SHM Function Selector provides quick access to <strong>108+ structural health monitoring functions</strong> 
      with intelligent parameter linking and comprehensive documentation.
    `;
    intro.style.cssText = `
      margin: 0 0 24px 0;
      color: #555;
      line-height: 1.5;
      text-align: center;
      font-size: 14px;
    `;

    // Keyboard shortcuts section
    const shortcutsSection = this.createHelpSection(
      '⌨️ Keyboard Shortcuts', 
      [
        '<kbd>Ctrl+Shift+F</kbd> - Open function browser',
        '<kbd>Ctrl+Shift+H</kbd> - Show function help for current cursor position',
        '<kbd>Ctrl+Shift+/</kbd> - Search functions'
      ]
    );

    // Basic usage section
    const usageSection = this.createHelpSection(
      '🚀 Basic Usage',
      [
        'Select a function from the dropdown to insert it into your notebook',
        'Use the 📖 button next to functions for detailed documentation',
        'Right-click on variables to automatically link them as function parameters',
        'Enable auto-insertion in settings for instant function placement',
        'Recently used functions appear at the top for quick access'
      ]
    );

    // Advanced features section  
    const advancedSection = this.createHelpSection(
      '⚡ Advanced Features',
      [
        '<strong>Smart Parameter Linking:</strong> Right-click variables to auto-populate function parameters',
        '<strong>Context-Aware Help:</strong> Use Ctrl+Shift+H while cursor is on a function name',
        '<strong>Function Categories:</strong> Browse functions organized by type (Core, Features, ML, etc.)',
        '<strong>Documentation Mode:</strong> Choose popup, inline, or sidebar documentation display',
        '<strong>Parameter Validation:</strong> Optional type checking for linked parameters'
      ]
    );

    // Tips section
    const tipsSection = this.createHelpSection(
      '💡 Pro Tips',
      [
        'Click the ⚙️ button to customize auto-insertion, shortcuts, and display preferences',
        'Use the search feature (Ctrl+Shift+/) to quickly find functions by name or category',
        'Function documentation includes examples, parameters, and return values',
        'Recently used functions are remembered across sessions',
        'Right-click sensitivity can be adjusted in settings for better parameter detection'
      ]
    );

    // Function categories section
    const categoriesSection = this.createHelpSection(
      '📂 Function Categories',
      [
        '<strong>Core:</strong> Signal processing, filtering, spectral analysis',
        '<strong>Features:</strong> Time series modeling, feature extraction',
        '<strong>Classification:</strong> Machine learning, outlier detection',
        '<strong>Modal:</strong> Modal analysis, structural dynamics',
        '<strong>Active Sensing:</strong> Guided wave analysis, sensor diagnostics',
        '<strong>Hardware:</strong> Data acquisition, sensor interfaces',
        '<strong>Plotting:</strong> Visualization utilities and interactive plots'
      ]
    );

    content.appendChild(title);
    content.appendChild(intro);
    content.appendChild(shortcutsSection);
    content.appendChild(usageSection);
    content.appendChild(advancedSection);
    content.appendChild(tipsSection);
    content.appendChild(categoriesSection);

    return content;
  }

  private createHelpSection(title: string, items: string[]): HTMLElement {
    const section = document.createElement('div');
    section.style.cssText = `
      margin-bottom: 20px;
      padding: 16px;
      background: #f8f9fa;
      border-radius: 6px;
      border-left: 4px solid #2196f3;
    `;

    const sectionTitle = document.createElement('h3');
    sectionTitle.innerHTML = title;
    sectionTitle.style.cssText = `
      margin: 0 0 12px 0;
      color: #1976d2;
      font-size: 16px;
      font-weight: 600;
    `;

    const list = document.createElement('ul');
    list.style.cssText = `
      margin: 0;
      padding-left: 20px;
      line-height: 1.6;
    `;

    items.forEach(item => {
      const listItem = document.createElement('li');
      listItem.innerHTML = item;
      listItem.style.cssText = `
        margin-bottom: 8px;
        color: #555;
        font-size: 14px;
      `;
      list.appendChild(listItem);
    });

    section.appendChild(sectionTitle);
    section.appendChild(list);

    return section;
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
  displayName?: string; // Human-readable name from verbose_call metadata
  type: string;
  value?: any;
  cellId: string;
  compatible: boolean;
  source?: string; // Function or expression that created the variable
}

class SHMContextMenuManager {
  private variables: Variable[] = [];
  private contextMenu: HTMLElement | null = null;
  private refreshCallback: (() => void) | null = null;
  
  setRefreshCallback(callback: () => void): void {
    this.refreshCallback = callback;
  }

  /**
   * Detect if right-click happened on a variable in cell output area ONLY
   */
  detectOutputVariable(event: MouseEvent, cell: any): string | null {
    const target = event.target as HTMLElement;
    
    // ONLY check if we're clicking on actual output area - be very restrictive
    if (target.closest('.jp-OutputArea')) {
      // Look for text content that looks like a variable name
      const outputText = target.textContent || '';
      
      // Simple pattern matching for variable names in output
      const variablePattern = /\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=/g;
      let variableMatches: string[] = [];
      let match;
      while ((match = variablePattern.exec(outputText)) !== null) {
        variableMatches.push(match[1]);
      }
      
      if (variableMatches.length > 0) {
        return variableMatches[0];
      }
      
      // Also check for standalone variable names that could be plottable
      const standalonePattern = /\b([a-zA-Z_][a-zA-Z0-9_]*)\b/g;
      let standaloneMatches: string[] = [];
      while ((match = standalonePattern.exec(outputText)) !== null) {
        standaloneMatches.push(match[1]);
      }
      
      // Filter out common non-variable words
      const excludeWords = ['array', 'dtype', 'shape', 'nan', 'inf', 'true', 'false', 'none', 'out', 'in'];
      
      for (const matchStr of standaloneMatches) {
        const word = matchStr.toLowerCase();
        if (!excludeWords.includes(word) && word.length > 1) {
          return matchStr;
        }
      }
      
      // If we're in output area but no specific variable found, offer to plot the last assignment
      const cellCode = cell.editor?.model?.sharedModel?.getSource() || '';
      return this.extractVariablesFromCodeForPlotting(cellCode);
    }
    
    // Return null if not in output area - let the original parameter detection handle it
    return null;
  }

  /**
   * Extract variable names from code (assignments) for plotting
   */
  extractVariablesFromCodeForPlotting(code: string): string | null {
    // Use regex to find variable assignments that span multiple lines
    // Pattern: variable(s) = anything (including multi-line function calls)
    
    // Remove comments first
    const codeWithoutComments = code.replace(/#[^\n]*/g, '');
    
    // Look for assignments that start at beginning of lines
    // This will match: "x, y, z = function(...)" even across multiple lines
    const assignmentRegex = /^([a-zA-Z_][a-zA-Z0-9_,\s]*)\s*=\s*[^=]/gm;
    
    let lastMatch = null;
    let match;
    
    // Find the LAST assignment in the code
    while ((match = assignmentRegex.exec(codeWithoutComments)) !== null) {
      lastMatch = match;
    }
    
    if (lastMatch) {
      const leftSide = lastMatch[1].trim();
      
      // Handle tuple assignments like "x, y, z = ..."
      if (leftSide.includes(',')) {
        const varNames = leftSide.split(',').map(v => v.trim()).filter(v => v.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/));
        return varNames.length > 0 ? varNames[0] : null;
      } else {
        // Single assignment like "x = ..."
        return leftSide.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/) ? leftSide : null;
      }
    }
    
    return null;
  }

  /**
   * Get all variables from the most recent assignment in code for plotting
   */
  getAllVariablesFromCodeForPlotting(code: string): string[] {
    // Remove comments first
    const codeWithoutComments = code.replace(/#[^\n]*/g, '');
    
    // Look for assignments that start at beginning of lines
    const assignmentRegex = /^([a-zA-Z_][a-zA-Z0-9_,\s]*)\s*=\s*[^=]/gm;
    
    let lastMatch = null;
    let match;
    
    // Find the LAST assignment in the code
    while ((match = assignmentRegex.exec(codeWithoutComments)) !== null) {
      lastMatch = match;
    }
    
    if (lastMatch) {
      const leftSide = lastMatch[1].trim();
      
      // Handle tuple assignments like "x, y, z = ..."
      if (leftSide.includes(',')) {
        return leftSide.split(',').map(v => v.trim()).filter(v => v.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/));
      } else {
        // Single assignment like "x = ..."
        return leftSide.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/) ? [leftSide] : [];
      }
    }
    
    return [];
  }

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
  async extractVariablesFromCells(notebook: any): Promise<void> {
    this.variables = [];
    
    // Collect all notebook cells for backend processing
    // Map code cell indices to actual notebook cell indices
    const cells = [];
    const codeCellToNotebookIndex = new Map<number, number>();
    let codeCellIndex = 0;
    
    for (let i = 0; i < notebook.model.cells.length; i++) {
      const cell = notebook.model.cells.get(i);
      if (cell.type === 'code') {
        cells.push({
          cell_type: 'code',
          source: cell.sharedModel.getSource()
        });
        codeCellToNotebookIndex.set(codeCellIndex, i);
        codeCellIndex++;
      }
    }
    
    try {
      // Use backend API for variable extraction with display names
      const response = await requestAPI<any>('variables', {
        method: 'POST',
        body: JSON.stringify({ cells })
      });
      
      console.log('🔍 DEBUG: Backend response:', response);
      
      // Convert backend response to frontend Variable format
      const backendVariables = Array.isArray(response) ? response : [];
      console.log(`🔍 DEBUG: Backend returned ${backendVariables.length} variables`);
      backendVariables.forEach((v: any) => {
        console.log(`  - Variable: ${v.name} from code cell ${v.cellIndex}, actual notebook cell ${codeCellToNotebookIndex.get(v.cellIndex)}, source: ${v.source}`);
      });
      
      this.variables = backendVariables.map((v: any, index: number) => {
        // Convert code cell index to actual notebook cell index
        const actualNotebookIndex = codeCellToNotebookIndex.get(v.cellIndex) ?? v.cellIndex;
        return {
          name: v.name,
          displayName: v.displayName,
          type: v.type || 'unknown',
          cellId: `cell-${actualNotebookIndex}`,
          compatible: false, // Will be set later based on context
          source: v.source
        };
      });
      
      console.log(`🔍 DEBUG: After mapping, this.variables has ${this.variables.length} items`);
      
    } catch (error) {
      console.warn('Backend variable extraction failed, falling back to frontend parsing:', error);
      
      // Fallback to original frontend parsing - use actual notebook indices
      for (let i = 0; i < notebook.model.cells.length; i++) {
        const cell = notebook.model.cells.get(i);
        if (cell.type === 'code') {
          const cellCode = cell.sharedModel.getSource();
          const cellId = `cell-${i}`;
          this.extractVariablesFromCode(cellCode, cellId);
        }
      }
    }
  }

  private extractVariablesFromCode(code: string, cellId: string): void {
    // Clean code by removing comments but preserve structure
    const cleanCode = code.replace(/#[^\n]*/g, '');
    
    // Match variable assignments that start at the beginning of lines (no indentation or minimal indentation)
    // This avoids matching function parameters like "    X=dataset" inside function calls
    const assignmentPattern = /^([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s*=\s*([^=].*?)(?=^[a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*\s*=|^#|\n\n|$)/gms;
    
    let match;
    while ((match = assignmentPattern.exec(cleanCode)) !== null) {
      const varsString = match[1].trim();
      const expression = match[2].trim();
      
      // Parse variable names (handle both single and tuple assignments)
      const varNames = varsString.includes(',') 
        ? varsString.split(',').map(v => v.trim())
        : [varsString];
      
      // Extract function name from potentially multi-line expression
      const source = this.extractFunctionNameFromExpression(expression);
      
      for (const varName of varNames) {
        // Skip common non-data variables
        if (varName && !['i', 'j', 'k', 'n', 'len', 'idx'].includes(varName)) {
          const variable: Variable = {
            name: varName,
            type: this.inferVariableType(cleanCode, varName),
            cellId: cellId,
            compatible: false, // Will be set based on parameter context
            source: source
          };
          
          console.log(`🔍 DEBUG: Extracted variable '${varName}' from ${cellId}:`, {
            name: variable.name,
            type: variable.type,
            source: variable.source,
            cellId: variable.cellId
          });
          
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
   * Extract function name from a potentially multi-line expression
   */
  private extractFunctionNameFromExpression(expression: string): string {
    // Remove all whitespace and newlines for easier parsing
    const cleanExpression = expression.replace(/\s+/g, ' ').trim();
    
    // Try to match function call patterns - now supports multi-line
    const funcMatch = cleanExpression.match(/^([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(/);
    
    if (funcMatch) {
      return funcMatch[1] + '()';
    }
    
    // If no function pattern found, return shortened expression
    return cleanExpression.substring(0, 30) + (cleanExpression.length > 30 ? '...' : '');
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
  async showContextMenu(
    event: MouseEvent, 
    parameterContext: ParameterContext, 
    notebook: any,
    currentCellIndex: number = -1,
    enableValidation: boolean = false
  ): Promise<void> {
    this.hideContextMenu();
    
    // Extract variables from all cells
    await this.extractVariablesFromCells(notebook);
    console.log(`🔍 DEBUG: Total variables extracted: ${this.variables.length}`);
    this.variables.forEach(v => console.log(`  - ${v.name} (${v.type}) from ${v.source} in ${v.cellId}`));
    
    // Filter to only show variables from cells before the current one
    console.log(`🔍 DEBUG: currentCellIndex parameter value: ${currentCellIndex}`);
    if (currentCellIndex >= 0) {
      console.log(`🔍 DEBUG: Filtering variables for currentCellIndex=${currentCellIndex}`);
      console.log(`🔍 DEBUG: Before filtering: ${this.variables.length} variables`);
      
      // Create filtered list with detailed logging
      const filteredVariables = [];
      for (const v of this.variables) {
        const cellNumber = parseInt(v.cellId.replace('cell-', ''));
        const shouldKeep = cellNumber < currentCellIndex;
        console.log(`  - ${v.name} in ${v.cellId} (cellNumber=${cellNumber}, currentCellIndex=${currentCellIndex}, will keep=${shouldKeep})`);
        if (shouldKeep) {
          filteredVariables.push(v);
        }
      }
      
      this.variables = filteredVariables;
      console.log(`🔍 DEBUG: After filtering for cells before ${currentCellIndex}: ${this.variables.length} variables`);
    } else {
      console.log(`🔍 DEBUG: currentCellIndex is ${currentCellIndex}, no filtering applied`);
    }
    
    // Extract input variable names from the current function call to exclude them
    const inputVariableNames = this.extractInputVariableNames(parameterContext);
    console.log(`🚫 Input variables to exclude: ${inputVariableNames.join(', ')}`);
    
    // Also exclude output variables from the same function type in previous cells
    const currentFunctionName = parameterContext.functionName;
    console.log(`🔍 Excluding outputs from previous ${currentFunctionName} calls`);
    
    // Filter out variables that are currently being used as inputs or are outputs from same function
    const beforeExclude = this.variables.length;
    this.variables = this.variables.filter(v => {
      // Exclude if it's a current input variable
      if (inputVariableNames.includes(v.name)) {
        console.log(`  - Excluding ${v.name}: currently used as input`);
        return false;
      }
      
      // Exclude if it's an output from the same function in a previous cell
      if (v.source && v.source.includes(currentFunctionName)) {
        console.log(`  - Excluding ${v.name}: output from previous ${currentFunctionName} call`);
        return false;
      }
      
      return true;
    });
    console.log(`🔍 DEBUG: After excluding input variables and same-function outputs: ${this.variables.length} variables (removed ${beforeExclude - this.variables.length})`);
    if (beforeExclude > this.variables.length) {
      console.log(`🔍 DEBUG: Kept ${this.variables.length} variables after filtering`);
    }
    
    console.log(`📊 Variables from cells before cell ${currentCellIndex}: ${this.variables.length}`);
    this.variables.forEach(v => {
      console.log(`  - ${v.name} (${v.type}) from ${v.source || 'unknown'} in ${v.cellId}`);
    });

    // Create context menu
    this.contextMenu = document.createElement('div');
    this.contextMenu.className = 'shm-context-menu';
    // Calculate responsive position and size
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const isMobile = viewportWidth < 768;
    
    let menuLeft = event.pageX;
    let menuTop = event.pageY;
    let menuWidth = isMobile ? 'min(300px, 80vw)' : '200px';
    let maxHeight = isMobile ? '50vh' : '300px';
    
    // Adjust position for mobile or if menu would go off-screen
    if (isMobile) {
      menuLeft = Math.max(10, Math.min(event.pageX, viewportWidth - 300));
      menuTop = Math.max(10, Math.min(event.pageY, viewportHeight - 200));
    } else {
      // Keep menu on screen for desktop
      if (menuLeft + 200 > viewportWidth) {
        menuLeft = viewportWidth - 210;
      }
      if (menuTop + 300 > viewportHeight) {
        menuTop = viewportHeight - 310;
      }
    }
    
    this.contextMenu.style.cssText = `
      position: fixed;
      left: ${menuLeft}px;
      top: ${menuTop}px;
      background: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: ${isMobile ? '13px' : '12px'};
      z-index: 10000;
      max-height: ${maxHeight};
      overflow-y: auto;
      min-width: ${menuWidth};
      max-width: 90vw;
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

    // Group variables by cell for color coding
    const groupByCellId = (vars: Variable[]) => {
      const groups = new Map<string, Variable[]>();
      vars.forEach(v => {
        if (!groups.has(v.cellId)) {
          groups.set(v.cellId, []);
        }
        groups.get(v.cellId)!.push(v);
      });
      return groups;
    };

    // Add all variables without sorting or categorization
    if (this.variables.length > 0) {
      const variableGroups = groupByCellId(this.variables);
      let cellColorIndex = 0;
      variableGroups.forEach((variables, cellId) => {
        variables.forEach(variable => {
          // Check if variable is compatible for determining styling
          const isCompatible = this.isVariableCompatible(variable, parameterContext);
          this.addVariableMenuItem(variable, parameterContext, notebook, isCompatible, enableValidation, cellColorIndex);
        });
        cellColorIndex++;
      });
    } else {
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
      console.log('🟢 CLOSHANDLER CALLED', {
        target: (e.target as HTMLElement).className,
        timestamp: Date.now()
      });
      const target = e.target as HTMLElement;
      // Check if click is on JupyterLab UI elements (toolbar buttons, cell controls, etc.)
      const isJupyterUIClick = target.closest('.jp-Toolbar') || 
                               target.closest('.jp-cell-toolbar') ||
                               target.closest('.jp-SideBar') ||
                               target.closest('.jp-MainAreaWidget-toolbar') ||
                               target.closest('[data-command]') ||
                               target.closest('.lm-MenuBar');
      
      // If clicking on JupyterLab UI, immediately remove handler and let the click through
      if (isJupyterUIClick) {
        console.log('🟢 CLOSHANDLER: Detected JupyterLab UI click, removing handler');
        document.removeEventListener('click', closeHandler);
        this.hideContextMenu();
        return;
      }
      
      // Otherwise, close menu if clicking outside
      if (!this.contextMenu?.contains(target)) {
        console.log('🟢 CLOSHANDLER: Click outside menu, closing');
        this.hideContextMenu();
        document.removeEventListener('click', closeHandler);
      }
    };
    
    // Use requestAnimationFrame instead of setTimeout to avoid interfering with UI operations
    console.log('🟢 CLOSHANDLER: Setting up with requestAnimationFrame');
    requestAnimationFrame(() => {
      console.log('🟢 CLOSHANDLER: Adding document click listener');
      document.addEventListener('click', closeHandler);
    });
  }

  /**
   * Extract variable names that are currently used as inputs in the function call
   */
  private extractInputVariableNames(parameterContext: ParameterContext): string[] {
    const inputVariables: string[] = [];
    
    try {
      // Get the current notebook cell to analyze the function call
      const activeCell = document.querySelector('.jp-Cell.jp-mod-active .jp-InputArea .jp-Editor') as HTMLElement;
      if (!activeCell) {
        console.log('DEBUG: No active cell found');
        return inputVariables;
      }
      
      // Get the CodeMirror instance to access the full code
      const codeMirrorDiv = activeCell.querySelector('.CodeMirror') as any;
      if (!codeMirrorDiv?.CodeMirror) {
        console.log('DEBUG: No CodeMirror instance found');
        return inputVariables;
      }
      
      const code = codeMirrorDiv.CodeMirror.getValue();
      console.log('DEBUG: Cell code:', code);
      
      // Find the function call that contains the current parameter
      const functionCallPattern = new RegExp(`${parameterContext.functionName}\\s*\\(([^)]+)\\)`, 'gs');
      const match = functionCallPattern.exec(code);
      
      if (match && match[1]) {
        const parametersText = match[1];
        console.log('DEBUG: Function parameters text:', parametersText);
        
        // Parse parameter values and extract variable names
        // Look for patterns like: param=variable_name or just variable_name
        const paramValuePattern = /(?:^\s*|,\s*)(?:\w+\s*=\s*)?([a-zA-Z_][a-zA-Z0-9_]*)/g;
        let valueMatch;
        
        while ((valueMatch = paramValuePattern.exec(parametersText)) !== null) {
          const potentialVariable = valueMatch[1];
          console.log('DEBUG: Found potential variable:', potentialVariable);
          
          // Skip common literals and keywords
          if (!['None', 'True', 'False', 'int', 'float', 'str', 'list', 'dict'].includes(potentialVariable) &&
              !potentialVariable.match(/^\d+$/)) {
            inputVariables.push(potentialVariable);
          }
        }
      } else {
        console.log('DEBUG: No function call match found for:', parameterContext.functionName);
      }
    } catch (error) {
      console.warn('Error extracting input variable names:', error);
    }
    
    console.log('DEBUG: Final input variables to exclude:', inputVariables);
    return [...new Set(inputVariables)]; // Remove duplicates
  }

  private addVariableMenuItem(
    variable: Variable, 
    parameterContext: ParameterContext, 
    notebook: any,
    isRecommended: boolean,
    enableValidation: boolean = false,
    cellColorIndex: number = 0
  ): void {
    const menuItem = document.createElement('div');
    menuItem.className = 'shm-context-menu-item';
    const isMobile = window.innerWidth < 768;
    
    // Determine background color based on cell color index - alternate between green and blue
    let backgroundColor: string;
    let hoverColor: string;
    backgroundColor = cellColorIndex % 2 === 0 ? '#f0fff0' : '#e0f7fa';
    hoverColor = cellColorIndex % 2 === 0 ? '#e8f5e8' : '#b2ebf2';
    
    menuItem.style.cssText = `
      padding: ${isMobile ? '12px 16px' : '8px 12px'};
      cursor: pointer;
      border-bottom: 1px solid #eee;
      transition: background 0.2s;
      background: ${backgroundColor};
      touch-action: manipulation;
      user-select: none;
      -webkit-tap-highlight-color: transparent;
    `;

    menuItem.innerHTML = `
      <div style="font-weight: bold; color: #333;">
        ${variable.displayName || variable.name}
      </div>
      <div style="font-size: 10px; color: #666;">
        ${variable.displayName && variable.displayName !== variable.name ? `${variable.name} • ` : ''}${variable.source && !variable.source.startsWith('Cell ') ? `${variable.source} • ` : ''}${variable.cellId.replace('cell-', 'Cell ')}
      </div>
    `;

    menuItem.addEventListener('mouseenter', () => {
      menuItem.style.background = hoverColor;
    });

    menuItem.addEventListener('mouseleave', () => {
      menuItem.style.background = backgroundColor;
    });

    menuItem.addEventListener('click', () => {
      this.linkParameterToVariable(variable, parameterContext, notebook, enableValidation);
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
    notebook: any,
    enableValidation: boolean = false
  ): void {
    const activeCell = notebook.activeCell;
    if (!activeCell) return;

    // Validate the parameter replacement before applying (if validation is enabled)
    if (enableValidation) {
      const validationResult = this.validateParameterReplacement(variable, parameterContext);
      if (!validationResult.isValid) {
        this.showValidationError(validationResult.error!);
        return;
      }
    }

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
    
    // Trigger refresh callback if set
    if (this.refreshCallback) {
      this.refreshCallback();
    }
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

  /**
   * Show plotting context menu for output variables
   */
  showPlottingContextMenu(
    event: MouseEvent, 
    variableName: string,
    consoleTracker: any
  ): void {
    this.hideContextMenu();
    
    // Create context menu
    this.contextMenu = document.createElement('div');
    this.contextMenu.className = 'shm-plotting-context-menu';
    
    // Calculate position
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    let menuLeft = event.pageX;
    let menuTop = event.pageY;
    
    // Adjust if menu would go off screen
    if (menuLeft + 250 > viewportWidth) {
      menuLeft = viewportWidth - 260;
    }
    if (menuTop + 200 > viewportHeight) {
      menuTop = viewportHeight - 210;
    }
    
    this.contextMenu.style.cssText = `
      position: fixed;
      left: ${menuLeft}px;
      top: ${menuTop}px;
      background: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      z-index: 10000;
      min-width: 220px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 13px;
    `;

    // Header
    const header = document.createElement('div');
    header.textContent = `📊 Plot: ${variableName}`;
    header.style.cssText = `
      padding: 10px 12px;
      background: #f5f5f5;
      border-bottom: 1px solid #ddd;
      font-weight: bold;
      color: #333;
    `;
    this.contextMenu.appendChild(header);

    // Plot options
    const plotOptions = [
      { label: '📈 Line Plot', code: `import matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\nplt.plot(${variableName})\nplt.title('${variableName}')\nplt.grid(True)\nplt.show()` },
      { label: '📊 Histogram', code: `import matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\nplt.hist(${variableName}, bins=30, alpha=0.7)\nplt.title('Histogram of ${variableName}')\nplt.xlabel('Values')\nplt.ylabel('Frequency')\nplt.grid(True, alpha=0.3)\nplt.show()` },
      { label: '🗺️ Heatmap (2D)', code: `import matplotlib.pyplot as plt\nimport numpy as np\nplt.figure(figsize=(10, 8))\nif ${variableName}.ndim == 2:\n    plt.imshow(${variableName}, cmap='viridis', aspect='auto')\n    plt.colorbar()\n    plt.title('Heatmap of ${variableName}')\nelse:\n    print("Variable must be 2D for heatmap")\nplt.show()` },
      { label: '📉 Scatter Plot', code: `import matplotlib.pyplot as plt\nimport numpy as np\nplt.figure(figsize=(10, 6))\nif ${variableName}.ndim == 1:\n    plt.scatter(range(len(${variableName})), ${variableName})\n    plt.xlabel('Index')\nelse:\n    if ${variableName}.shape[1] >= 2:\n        plt.scatter(${variableName}[:, 0], ${variableName}[:, 1])\n        plt.xlabel('Column 0')\n        plt.ylabel('Column 1')\n    else:\n        print("Need at least 2 columns for scatter plot")\nplt.title('Scatter Plot of ${variableName}')\nplt.grid(True, alpha=0.3)\nplt.show()` }
    ];

    plotOptions.forEach(option => {
      const menuItem = document.createElement('div');
      menuItem.textContent = option.label;
      menuItem.style.cssText = `
        padding: 8px 12px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
        transition: background-color 0.2s;
      `;

      menuItem.addEventListener('mouseenter', () => {
        menuItem.style.background = '#f0f0f0';
      });

      menuItem.addEventListener('mouseleave', () => {
        menuItem.style.background = '';
      });

      menuItem.addEventListener('click', () => {
        this.executeInConsole(option.code, consoleTracker);
        this.hideContextMenu();
      });

      this.contextMenu.appendChild(menuItem);
    });

    document.body.appendChild(this.contextMenu);

    // Close menu on outside click
    const closeHandler = (e: MouseEvent) => {
      console.log('🟢 CLOSHANDLER CALLED', {
        target: (e.target as HTMLElement).className,
        timestamp: Date.now()
      });
      const target = e.target as HTMLElement;
      // Check if click is on JupyterLab UI elements (toolbar buttons, cell controls, etc.)
      const isJupyterUIClick = target.closest('.jp-Toolbar') || 
                               target.closest('.jp-cell-toolbar') ||
                               target.closest('.jp-SideBar') ||
                               target.closest('.jp-MainAreaWidget-toolbar') ||
                               target.closest('[data-command]') ||
                               target.closest('.lm-MenuBar');
      
      // If clicking on JupyterLab UI, immediately remove handler and let the click through
      if (isJupyterUIClick) {
        console.log('🟢 CLOSHANDLER: Detected JupyterLab UI click, removing handler');
        document.removeEventListener('click', closeHandler);
        this.hideContextMenu();
        return;
      }
      
      // Otherwise, close menu if clicking outside
      if (!this.contextMenu?.contains(target)) {
        console.log('🟢 CLOSHANDLER: Click outside menu, closing');
        this.hideContextMenu();
        document.removeEventListener('click', closeHandler);
      }
    };
    
    // Use requestAnimationFrame instead of setTimeout to avoid interfering with UI operations
    console.log('🟢 CLOSHANDLER: Setting up with requestAnimationFrame');
    requestAnimationFrame(() => {
      console.log('🟢 CLOSHANDLER: Adding document click listener');
      document.addEventListener('click', closeHandler);
    });
  }

  /**
   * Execute code in the console associated with the current notebook
   */
  async executeInConsole(code: string, consoleTracker: any): Promise<void> {
    try {
      // Find or create a console for the current notebook
      let console = consoleTracker.currentWidget;
      
      if (!console) {
        // No console open, show notification
        const notification = document.createElement('div');
        notification.textContent = '📱 Please open a console first (File → New → Console)';
        notification.style.cssText = `
          position: fixed;
          top: 20px;
          right: 20px;
          background: #ff9800;
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
        }, 4000);
        return;
      }

      // Execute the code in the console
      await console.console.inject(code, false);
      
      // Show success notification
      const notification = document.createElement('div');
      notification.textContent = '✅ Plot command sent to console';
      notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #4caf50;
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
      
    } catch (error) {
      console.error('Error executing code in console:', error);
      
      // Show error notification
      const notification = document.createElement('div');
      notification.textContent = '❌ Error sending to console';
      notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #f44336;
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
  }

  /**
   * Show plotting context menu for multiple variables
   */
  showMultiVariablePlottingMenu(
    event: MouseEvent, 
    variables: string[],
    consoleTracker: any
  ): void {
    this.hideContextMenu();
    
    // Create context menu
    this.contextMenu = document.createElement('div');
    this.contextMenu.className = 'shm-multi-plotting-context-menu';
    
    // Calculate position
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    let menuLeft = event.pageX;
    let menuTop = event.pageY;
    
    // Adjust if menu would go off screen
    if (menuLeft + 300 > viewportWidth) {
      menuLeft = viewportWidth - 310;
    }
    if (menuTop + 250 > viewportHeight) {
      menuTop = viewportHeight - 260;
    }
    
    this.contextMenu.style.cssText = `
      position: fixed;
      left: ${menuLeft}px;
      top: ${menuTop}px;
      background: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      z-index: 10000;
      min-width: 250px;
      max-height: 400px;
      overflow-y: auto;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 13px;
    `;

    // Header
    const header = document.createElement('div');
    header.textContent = `📊 Select Variable to Plot`;
    header.style.cssText = `
      padding: 10px 12px;
      background: #f5f5f5;
      border-bottom: 1px solid #ddd;
      font-weight: bold;
      color: #333;
      position: sticky;
      top: 0;
    `;
    this.contextMenu.appendChild(header);

    // Variable list
    variables.forEach((variable, index) => {
      const variableItem = document.createElement('div');
      variableItem.textContent = `${index + 1}. ${variable}`;
      variableItem.style.cssText = `
        padding: 8px 12px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
        transition: background-color 0.2s;
        display: flex;
        align-items: center;
      `;

      variableItem.addEventListener('mouseenter', () => {
        variableItem.style.background = '#f0f8ff';
      });

      variableItem.addEventListener('mouseleave', () => {
        variableItem.style.background = '';
      });

      variableItem.addEventListener('click', () => {
        this.hideContextMenu();
        // Show individual plotting menu for selected variable
        setTimeout(() => {
          this.showPlottingContextMenu(event, variable, consoleTracker);
        }, 100);
      });

      this.contextMenu.appendChild(variableItem);
    });

    document.body.appendChild(this.contextMenu);

    // Close menu on outside click
    const closeHandler = (e: MouseEvent) => {
      console.log('🟢 CLOSHANDLER CALLED', {
        target: (e.target as HTMLElement).className,
        timestamp: Date.now()
      });
      const target = e.target as HTMLElement;
      // Check if click is on JupyterLab UI elements (toolbar buttons, cell controls, etc.)
      const isJupyterUIClick = target.closest('.jp-Toolbar') || 
                               target.closest('.jp-cell-toolbar') ||
                               target.closest('.jp-SideBar') ||
                               target.closest('.jp-MainAreaWidget-toolbar') ||
                               target.closest('[data-command]') ||
                               target.closest('.lm-MenuBar');
      
      // If clicking on JupyterLab UI, immediately remove handler and let the click through
      if (isJupyterUIClick) {
        console.log('🟢 CLOSHANDLER: Detected JupyterLab UI click, removing handler');
        document.removeEventListener('click', closeHandler);
        this.hideContextMenu();
        return;
      }
      
      // Otherwise, close menu if clicking outside
      if (!this.contextMenu?.contains(target)) {
        console.log('🟢 CLOSHANDLER: Click outside menu, closing');
        this.hideContextMenu();
        document.removeEventListener('click', closeHandler);
      }
    };
    
    // Use requestAnimationFrame instead of setTimeout to avoid interfering with UI operations
    console.log('🟢 CLOSHANDLER: Setting up with requestAnimationFrame');
    requestAnimationFrame(() => {
      console.log('🟢 CLOSHANDLER: Adding document click listener');
      document.addEventListener('click', closeHandler);
    });
  }

  /**
   * Validate parameter replacement before applying
   */
  validateParameterReplacement(variable: Variable, parameterContext: ParameterContext): { isValid: boolean; error?: string } {
    // Basic type compatibility check
    const paramName = parameterContext.parameterName.toLowerCase();
    const varType = variable.type.toLowerCase();
    
    // Array/data parameters should receive array-like variables
    if (['data', 'features', 'input_data', 'signals', 'x', 'y'].includes(paramName)) {
      if (!['numpy.ndarray', 'pandas.dataframe', 'list', 'tuple'].includes(varType)) {
        return {
          isValid: false,
          error: `Parameter "${parameterContext.parameterName}" expects array data, but "${variable.name}" is of type ${variable.type}`
        };
      }
    }
    
    // Frequency parameters should receive numeric variables
    if (['fs', 'sampling_rate', 'freq', 'frequency'].includes(paramName)) {
      if (!['int', 'float', 'numpy.float64', 'numpy.int64'].includes(varType) && !variable.name.toLowerCase().includes('freq') && !variable.name.toLowerCase().includes('fs')) {
        return {
          isValid: false,
          error: `Parameter "${parameterContext.parameterName}" expects a frequency value, but "${variable.name}" may not be a frequency`
        };
      }
    }
    
    // Order parameters should receive integer variables
    if (['order', 'n_components', 'ar_order'].includes(paramName)) {
      if (!['int', 'numpy.int64'].includes(varType) && isNaN(parseInt(variable.name))) {
        return {
          isValid: false,
          error: `Parameter "${parameterContext.parameterName}" expects an integer value, but "${variable.name}" is of type ${variable.type}`
        };
      }
    }
    
    // Model parameters should receive dict/tuple variables
    if (paramName.includes('model')) {
      if (!['dict', 'tuple', 'unknown'].includes(varType)) {
        return {
          isValid: false,
          error: `Parameter "${parameterContext.parameterName}" expects a model object, but "${variable.name}" is of type ${variable.type}`
        };
      }
    }
    
    return { isValid: true };
  }

  /**
   * Show validation error to user
   */
  showValidationError(error: string): void {
    const notification = document.createElement('div');
    notification.textContent = `⚠️ Validation Error: ${error}`;
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #ff5722;
      color: white;
      padding: 12px 16px;
      border-radius: 4px;
      z-index: 10000;
      font-family: monospace;
      font-size: 12px;
      max-width: 400px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 5000);
  }

  /**
   * Enhanced parameter validation with specific rules
   */
  validateParameterValue(value: string, parameterName: string, validation: any[]): { isValid: boolean; error?: string } {
    if (!validation || validation.length === 0) {
      return { isValid: true };
    }
    
    for (const rule of validation) {
      if (rule.type === 'range') {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          return {
            isValid: false,
            error: `Parameter "${parameterName}" must be a number for range validation`
          };
        }
        if (numValue < rule.min || numValue > rule.max) {
          return {
            isValid: false,
            error: `Parameter "${parameterName}" must be between ${rule.min} and ${rule.max}, got ${numValue}`
          };
        }
      } else if (rule.type === 'choice') {
        const cleanValue = value.replace(/['"]/g, '');
        if (!rule.options.includes(cleanValue)) {
          return {
            isValid: false,
            error: `Parameter "${parameterName}" must be one of: ${rule.options.join(', ')}, got "${cleanValue}"`
          };
        }
      } else if (rule.type === 'file_format') {
        const cleanValue = value.replace(/['"]/g, '');
        const hasValidExtension = rule.formats.some((fmt: string) => cleanValue.endsWith(fmt));
        if (!hasValidExtension) {
          return {
            isValid: false,
            error: `Parameter "${parameterName}" file must have one of these extensions: ${rule.formats.join(', ')}`
          };
        }
      }
    }
    
    return { isValid: true };
  }
}