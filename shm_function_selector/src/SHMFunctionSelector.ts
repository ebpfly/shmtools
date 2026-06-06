// Function browser dropdown component for the SHM Function Selector extension.
//
// Provides the categorized function dropdown, search/filter, documentation
// popups, settings/help panels and code-snippet generation/insertion.
//
// Extracted from index.ts as part of the modular refactor (issue #30).

import { JupyterFrontEnd } from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';
import { MessageLoop } from '@lumino/messaging';
import { Widget } from '@lumino/widgets';

import { requestAPI } from './serverAPI';
import { SHMFunction, CategoryNode } from './types';

export class SHMFunctionSelector {
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
