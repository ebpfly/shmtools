// Entry point for the SHM Function Selector JupyterLab extension.
//
// This file wires the plugin together; the heavy lifting lives in dedicated
// modules:
//   - SHMFunctionSelector    - function browser dropdown UI
//   - SHMContextMenuManager  - right-click parameter linking / plotting menus
//   - keyboardShortcuts      - keyboard shortcuts and search/quick-select
//   - types                  - shared interfaces
//
// Refactored from a single 5,500+ line module as part of issue #30.

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { INotebookTracker } from '@jupyterlab/notebook';
import { IConsoleTracker } from '@jupyterlab/console';

import { SHMFunctionSelector } from './SHMFunctionSelector';
import { SHMContextMenuManager } from './SHMContextMenuManager';
import { setupKeyboardShortcuts } from './keyboardShortcuts';

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
