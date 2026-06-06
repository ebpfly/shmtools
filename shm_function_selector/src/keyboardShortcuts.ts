// Keyboard shortcut registration and associated quick-select / search dialogs
// for the SHM Function Selector extension.
//
// Extracted from index.ts as part of the modular refactor (issue #30).

import { JupyterFrontEnd } from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';

import { SHMFunctionSelector } from './SHMFunctionSelector';
import { SHMContextMenuManager } from './SHMContextMenuManager';

export function setupKeyboardShortcuts(
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
