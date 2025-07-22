/**
 * SHM Function Selector Jupyter Notebook Extension
 * 
 * Provides:
 * 1. Toolbar dropdown for function selection
 * 2. Right-click context menus for parameter linking  
 * 3. Auto-populated default values
 */

// Support both old require.js and new module systems
if (typeof define === 'function' && define.amd) {
    // AMD/RequireJS (older notebooks)
    define([
        'base/js/namespace',
        'base/js/events', 
        'base/js/utils',
        'notebook/js/codecell'
    ], function(Jupyter, events, utils, codecell) {
        return createSHMExtension(Jupyter, events, utils, codecell);
    });
} else {
    // Modern notebooks - load directly
    (function() {
        // Wait for Jupyter to be available
        function waitForJupyter() {
            if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {
                createSHMExtension(Jupyter);
            } else {
                setTimeout(waitForJupyter, 100);
            }
        }
        waitForJupyter();
    })();
}

function createSHMExtension(Jupyter, events, utils, codecell) {
    
    var SHMFunctionSelector = {
        
        // Extension configuration
        functions: [],
        variables: [],
        parsedVariables: [],
        
        // Initialize the extension
        load_ipython_extension: function() {
            console.log('Loading SHM Function Selector extension');
            
            this.setupToolbar();
            this.setupContextMenu();
            this.loadSHMFunctions();
            
            console.log('SHM Function Selector extension loaded');
        },
        
        // Setup toolbar dropdown
        setupToolbar: function() {
            var that = this;
            
            // Create dropdown button
            var dropdown = $('<div class="btn-group">' +
                '<button type="button" class="btn btn-default dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">' +
                'SHM Functions <span class="caret"></span>' +
                '</button>' +
                '<ul class="dropdown-menu" id="shm-function-dropdown">' +
                '<li><a href="#" data-loading="true">Loading functions...</a></li>' +
                '</ul>' +
                '</div>');
            
            // Add to toolbar
            Jupyter.toolbar.add_buttons_group([{
                id: 'shm-function-selector',
                label: 'SHM Function Selector', 
                icon: 'fa-plus',
                callback: function() {
                    // Dropdown is handled by Bootstrap
                }
            }]);
            
            // Replace the button with our dropdown
            $('#shm-function-selector').parent().replaceWith(dropdown);
        },
        
        // Load SHM functions from server
        loadSHMFunctions: function() {
            var that = this;
            
            // Use modern fetch if available, fallback to jQuery
            var baseUrl = Jupyter.notebook ? Jupyter.notebook.base_url : '';
            var url = baseUrl + 'shm_extension/functions';
            
            if (typeof fetch !== 'undefined') {
                fetch(url)
                    .then(response => response.json())
                    .then(data => {
                        that.functions = data;
                        that.populateDropdown();
                    })
                    .catch(error => {
                        console.error('Failed to load SHM functions:', error);
                        that.showErrorInDropdown('Failed to load functions');
                    });
            } else if (typeof $ !== 'undefined') {
                $.ajax({
                    url: url,
                    method: 'GET',
                    success: function(data) {
                        that.functions = typeof data === 'string' ? JSON.parse(data) : data;
                        that.populateDropdown();
                    },
                    error: function(xhr, status, error) {
                        console.error('Failed to load SHM functions:', error);
                        that.showErrorInDropdown('Failed to load functions');
                    }
                });
            } else {
                console.error('Neither fetch nor jQuery available');
                that.showErrorInDropdown('Browser compatibility issue');
            }
        },
        
        // Populate dropdown with functions organized by category
        populateDropdown: function() {
            var dropdown = $('#shm-function-dropdown');
            dropdown.empty();
            
            if (this.functions.length === 0) {
                dropdown.append('<li><a href="#" data-disabled="true">No functions available</a></li>');
                return;
            }
            
            // Group functions by category
            var categories = {};
            this.functions.forEach(function(func) {
                if (!categories[func.category]) {
                    categories[func.category] = [];
                }
                categories[func.category].push(func);
            });
            
            // Add categories to dropdown
            var that = this;
            Object.keys(categories).forEach(function(category) {
                // Add category header
                dropdown.append('<li class="dropdown-header">' + category + '</li>');
                
                // Add functions in this category
                categories[category].forEach(function(func) {
                    var item = $('<li><a href="#" data-function="' + func.name + '">' +
                        func.display_name + '</a></li>');
                    
                    item.find('a').click(function(e) {
                        e.preventDefault();
                        that.insertFunction(func);
                    });
                    
                    dropdown.append(item);
                });
                
                // Add separator
                dropdown.append('<li role="separator" class="divider"></li>');
            });
        },
        
        // Insert selected function into notebook
        insertFunction: function(func) {
            console.log('Inserting function:', func.name);
            
            // Generate function call code
            var code = this.generateFunctionCall(func);
            
            // Insert into current cell or create new cell
            var cell = Jupyter.notebook.get_selected_cell();
            
            if (cell && cell.cell_type === 'code' && cell.get_text().trim() === '') {
                // Use current empty cell
                cell.set_text(code);
            } else {
                // Create new cell
                var new_cell = Jupyter.notebook.insert_cell_below('code');
                new_cell.set_text(code);
                Jupyter.notebook.select_next();
            }
        },
        
        // Generate function call code with parameters
        generateFunctionCall: function(func) {
            var lines = [];
            
            // Add comment with function description
            if (func.description) {
                lines.push('# ' + func.description);
            }
            
            // Generate function call
            var params = [];
            func.parameters.forEach(function(param) {
                if (param.name === 'self') return; // Skip self parameter
                
                var paramStr = param.name + '=';
                
                if (param.default) {
                    // Use default value
                    paramStr += param.default;
                } else {
                    // Placeholder for required parameter
                    paramStr += 'None  # TODO: Set ' + param.name;
                }
                
                params.push(paramStr);
            });
            
            // Determine output variables (simplified)
            var outputs = 'result';
            if (func.name.includes('pca') || func.name.includes('ar_model')) {
                outputs = 'features, model';
            }
            
            var functionCall = outputs + ' = shmtools.' + func.name + '(';
            if (params.length > 0) {
                functionCall += '\\n    ' + params.join(',\\n    ') + '\\n';
            }
            functionCall += ')';
            
            lines.push(functionCall);
            
            return lines.join('\\n');
        },
        
        // Setup right-click context menu
        setupContextMenu: function() {
            var that = this;
            
            // Add context menu to code cells
            $(document).on('contextmenu', '.code_cell .input_area', function(e) {
                var cell = Jupyter.notebook.get_selected_cell();
                if (!cell || cell.cell_type !== 'code') return;
                
                var cursor = cell.code_mirror.getCursor();
                var parameterContext = that.detectParameterContext(cell, cursor);
                
                if (parameterContext) {
                    that.showParameterContextMenu(e, cursor, parameterContext);
                }
            });
        },
        
        // Enhanced parameter context detection
        detectParameterContext: function(cell, cursor) {
            var code = cell.get_text();
            var lines = code.split('\n');
            var line = lines[cursor.line] || '';
            
            // Get character position in line
            var ch = cursor.ch;
            
            // Check if cursor is on or near a parameter assignment
            var parameterInfo = this.parseParameterAtPosition(line, ch);
            
            if (!parameterInfo) {
                // Also check adjacent lines for multi-line function calls
                var context = this.getMultilineContext(lines, cursor.line, cursor.ch);
                if (context) {
                    parameterInfo = this.parseParameterInContext(context, ch);
                }
            }
            
            return parameterInfo;
        },
        
        // Parse parameter at specific position in line
        parseParameterAtPosition: function(line, ch) {
            // Patterns to match parameter assignments
            var patterns = [
                // Named parameter: param=value
                {
                    regex: /([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,)]*)/g,
                    type: 'named_param'
                },
                // Function call with placeholder
                {
                    regex: /(shmtools\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g,
                    type: 'function_call'
                },
                // TODO comments for parameters
                {
                    regex: /#\s*TODO:\s*[Ss]et\s+([a-zA-Z_][a-zA-Z0-9_]*)/g,
                    type: 'todo_param'
                }
            ];
            
            for (var i = 0; i < patterns.length; i++) {
                var pattern = patterns[i];
                var match;
                
                while ((match = pattern.regex.exec(line)) !== null) {
                    var startPos = match.index;
                    var endPos = startPos + match[0].length;
                    
                    // Check if cursor is within this match
                    if (ch >= startPos && ch <= endPos) {
                        if (pattern.type === 'named_param') {
                            var paramName = match[1];
                            var paramValue = match[2].trim();
                            
                            // Only show context menu for placeholder values
                            if (paramValue === 'None' || paramValue.includes('TODO') || paramValue === '') {
                                return {
                                    parameterName: paramName,
                                    parameterValue: paramValue,
                                    startPosition: startPos,
                                    endPosition: endPos,
                                    replacementStart: match.index + match[1].length + 1, // After 'param='
                                    replacementEnd: endPos,
                                    type: 'named_parameter'
                                };
                            }
                        } else if (pattern.type === 'todo_param') {
                            return {
                                parameterName: match[1],
                                type: 'todo_parameter',
                                startPosition: startPos,
                                endPosition: endPos
                            };
                        }
                    }
                }
                // Reset regex lastIndex for next iteration
                pattern.regex.lastIndex = 0;
            }
            
            return null;
        },
        
        // Get multi-line context for function calls
        getMultilineContext: function(lines, lineIndex, ch) {
            // Look for function calls that span multiple lines
            var startLine = lineIndex;
            var endLine = lineIndex;
            
            // Find the start of function call (look backward for opening parenthesis)
            for (var i = lineIndex; i >= Math.max(0, lineIndex - 5); i--) {
                if (lines[i].includes('shmtools.') && lines[i].includes('(')) {
                    startLine = i;
                    break;
                }
            }
            
            // Find the end of function call (look forward for closing parenthesis)
            var parenCount = 0;
            for (var i = startLine; i < Math.min(lines.length, lineIndex + 5); i++) {
                var line = lines[i];
                for (var j = 0; j < line.length; j++) {
                    if (line[j] === '(') parenCount++;
                    if (line[j] === ')') parenCount--;
                    if (parenCount === 0 && i >= startLine) {
                        endLine = i;
                        break;
                    }
                }
                if (parenCount === 0) break;
            }
            
            if (startLine < lineIndex && endLine >= lineIndex) {
                return {
                    lines: lines.slice(startLine, endLine + 1),
                    startLine: startLine,
                    endLine: endLine,
                    currentLine: lineIndex
                };
            }
            
            return null;
        },
        
        // Parse parameter in multi-line context
        parseParameterInContext: function(context, ch) {
            var fullText = context.lines.join('\n');
            var currentLineStart = 0;
            
            // Calculate character position in full context
            for (var i = 0; i < context.currentLine - context.startLine; i++) {
                currentLineStart += context.lines[i].length + 1; // +1 for newline
            }
            var absoluteCh = currentLineStart + ch;
            
            // Use the single-line parser on the full context
            return this.parseParameterAtPosition(fullText, absoluteCh);
        },
        
        // Show context menu for parameter linking
        showParameterContextMenu: function(event, cursor, parameterContext) {
            event.preventDefault();
            
            // Parse current notebook variables
            this.parseNotebookVariables();
            
            // Create enhanced context menu
            var menu = $('<div class="shm-context-menu" style="' +
                'position: absolute; ' +
                'z-index: 1000; ' +
                'background: white; ' +
                'border: 1px solid #ccc; ' +
                'border-radius: 3px; ' +
                'padding: 8px; ' +
                'box-shadow: 0 2px 10px rgba(0,0,0,0.2); ' +
                'max-height: 300px; ' +
                'overflow-y: auto; ' +
                'min-width: 250px; ' +
                'font-family: monospace; ' +
                'font-size: 12px;' +
                '">' +
                '<div style="font-weight: bold; color: #333; border-bottom: 1px solid #eee; padding-bottom: 4px; margin-bottom: 6px;">' +
                'Link parameter: ' + parameterContext.parameterName +
                '</div>' +
                '</div>');
            
            var that = this;
            
            // Store parameter context for later use
            this.currentParameterContext = parameterContext;
            
            // Add variables to menu with smart filtering and grouping
            if (this.parsedVariables.length === 0) {
                menu.append('<div style="color: #999; font-style: italic; padding: 4px;">No variables found</div>');
            } else {
                // Get current cell index to only show variables from previous cells
                var currentCellIndex = Jupyter.notebook.get_selected_index();
                var availableVars = this.parsedVariables.filter(function(v) {
                    return v.cellIndex < currentCellIndex;
                });
                
                if (availableVars.length === 0) {
                    menu.append('<div style="color: #999; font-style: italic; padding: 4px;">No variables from previous cells</div>');
                } else {
                    // Group variables by compatibility
                    var compatibleVars = [];
                    var otherVars = [];
                    
                    availableVars.forEach(function(variable) {
                        if (that.isVariableCompatible(variable, parameterContext)) {
                            compatibleVars.push(variable);
                        } else {
                            otherVars.push(variable);
                        }
                    });
                    
                    // Add compatible variables first (with visual emphasis)
                    if (compatibleVars.length > 0) {
                        menu.append('<div style="color: #388e3c; font-size: 11px; font-weight: bold; margin: 4px 0;">Recommended:</div>');
                        compatibleVars.forEach(function(variable) {
                            var item = that.createVariableMenuItem(variable, true);
                            menu.append(item);
                        });
                    }
                    
                    // Add other variables (dimmed)
                    if (otherVars.length > 0) {
                        if (compatibleVars.length > 0) {
                            menu.append('<div style="color: #757575; font-size: 11px; font-weight: bold; margin: 4px 0; border-top: 1px solid #eee; padding-top: 4px;">Other variables:</div>');
                        }
                        otherVars.forEach(function(variable) {
                            var item = that.createVariableMenuItem(variable, false);
                            menu.append(item);
                        });
                    }
                }
            }
            
            // Position menu
            menu.css({
                left: event.pageX + 'px',
                top: event.pageY + 'px'
            });
            
            // Handle clicks
            menu.find('a').click(function(e) {
                e.preventDefault();
                var variable = $(this).data('variable');
                that.linkParameterToVariable(cursor, variable);
                menu.remove();
            });
            
            // Add to document
            $('body').append(menu);
            
            // Remove on click outside
            $(document).one('click', function() {
                menu.remove();
            });
        },
        
        // Check if a variable is compatible with a parameter
        isVariableCompatible: function(variable, parameterContext) {
            var paramName = parameterContext.parameterName.toLowerCase();
            var varType = variable.type.toLowerCase();
            
            // Data parameter compatibility rules
            if (paramName === 'data' || paramName === 'features' || paramName === 'input_data') {
                return varType.includes('numpy') || varType.includes('array') || varType.includes('ndarray');
            }
            
            // Model parameter compatibility
            if (paramName === 'model' || paramName.includes('model')) {
                return varType.includes('dict') || varType.includes('tuple') || varType.includes('unknown');
            }
            
            // Frequency/sampling rate parameters
            if (paramName === 'fs' || paramName === 'sampling_rate' || paramName === 'freq') {
                return varType.includes('float') || varType.includes('int') || varType.includes('number');
            }
            
            // Order parameters
            if (paramName === 'order' || paramName === 'n_components' || paramName.includes('_order')) {
                return varType.includes('int') || varType.includes('float');
            }
            
            // Channel-related parameters
            if (paramName === 'channels' || paramName === 'channel_names') {
                return varType.includes('list') || varType.includes('array');
            }
            
            // Default: arrays are generally compatible, basic types less so
            if (varType.includes('numpy') || varType.includes('array')) return true;
            if (varType.includes('tuple') && (paramName.includes('model') || paramName.includes('result'))) return true;
            
            return false;
        },
        
        // Create a menu item for a variable
        createVariableMenuItem: function(variable, isRecommended) {
            var that = this;
            var typeColor = isRecommended ? '#388e3c' : '#757575';
            var opacity = isRecommended ? '1.0' : '0.7';
            
            // Format variable display with type information
            var typeInfo = this.formatVariableType(variable);
            var sourceInfo = variable.source || ('Cell ' + (variable.cellIndex + 1));
            
            var item = $('<div style="margin: 2px 0;">' +
                '<a href="#" ' +
                'data-variable="' + variable.name + '" ' +
                'data-type="' + variable.type + '" ' +
                'style="display: block; padding: 4px 6px; text-decoration: none; ' +
                'border: 1px solid transparent; border-radius: 2px; opacity: ' + opacity + ';">' +
                '<div style="color: #333; font-weight: bold;">' + variable.name + '</div>' +
                '<div style="color: ' + typeColor + '; font-size: 10px;">' + typeInfo + ' • ' + sourceInfo + '</div>' +
                '</a>' +
                '</div>');
            
            // Enhanced hover effects
            item.find('a')
                .hover(
                    function() { 
                        $(this).css({
                            'background-color': '#f5f5f5',
                            'border-color': '#ddd'
                        });
                    },
                    function() { 
                        $(this).css({
                            'background-color': 'transparent',
                            'border-color': 'transparent'
                        });
                    }
                )
                .click(function(e) {
                    e.preventDefault();
                    var variableName = $(this).data('variable');
                    that.linkParameterToVariable(that.currentParameterContext, variableName);
                    $('.shm-context-menu').remove();
                });
            
            return item;
        },
        
        // Format variable type information for display
        formatVariableType: function(variable) {
            var type = variable.type;
            
            // Simplify common type names
            if (type === 'numpy.ndarray' || type === 'ndarray') {
                return 'array';
            } else if (type === 'unknown') {
                return '?';
            } else if (type.length > 12) {
                return type.substring(0, 10) + '...';
            }
            
            return type;
        },
        
        // Enhanced parameter linking with precise positioning
        linkParameterToVariable: function(parameterContext, variableName) {
            var cell = Jupyter.notebook.get_selected_cell();
            
            if (parameterContext.type === 'named_parameter') {
                // Use precise replacement positions
                var line = cell.code_mirror.getLine(cell.code_mirror.getCursor().line);
                var beforeParam = line.substring(0, parameterContext.replacementStart);
                var afterParam = line.substring(parameterContext.replacementEnd);
                
                // Construct new line with proper spacing
                var newValue = variableName;
                if (parameterContext.parameterValue.includes('TODO')) {
                    // Remove TODO comment when replacing
                    afterParam = afterParam.replace(/\s*#.*$/, '');
                }
                
                var newLine = beforeParam + newValue + afterParam;
                
                cell.code_mirror.replaceRange(
                    newLine, 
                    {line: cell.code_mirror.getCursor().line, ch: 0}, 
                    {line: cell.code_mirror.getCursor().line, ch: line.length}
                );
            } else {
                // Fallback to simple replacement for other types
                var cursor = cell.code_mirror.getCursor();
                var line = cell.code_mirror.getLine(cursor.line);
                var newLine = line.replace(/=\s*None(\s*#.*)?$/, '=' + variableName);
                
                cell.code_mirror.replaceRange(
                    newLine, 
                    {line: cursor.line, ch: 0}, 
                    {line: cursor.line, ch: line.length}
                );
            }
        },
        
        // Show error in dropdown
        showErrorInDropdown: function(message) {
            var dropdown = $('#shm-function-dropdown');
            dropdown.empty();
            dropdown.append('<li><a href="#" data-error="true" style="color: red;">' + message + '</a></li>');
        },
        
        // Parse notebook cells to extract variable assignments
        parseNotebookVariables: function() {
            var that = this;
            that.parsedVariables = [];
            
            // Get all code cells from notebook
            var cells = Jupyter.notebook.get_cells();
            
            cells.forEach(function(cell, index) {
                if (cell.cell_type === 'code') {
                    var code = cell.get_text();
                    var cellVariables = that.extractVariablesFromCode(code, index);
                    that.parsedVariables = that.parsedVariables.concat(cellVariables);
                }
            });
            
            console.log('Parsed variables:', that.parsedVariables);
            return that.parsedVariables;
        },
        
        // Alternative: parse variables using server-side handler
        parseNotebookVariablesServer: function(callback) {
            var that = this;
            
            // Collect notebook cell data
            var cells = Jupyter.notebook.get_cells().map(function(cell, index) {
                return {
                    cell_type: cell.cell_type,
                    source: cell.get_text()
                };
            });
            
            var baseUrl = Jupyter.notebook ? Jupyter.notebook.base_url : '';
            var url = baseUrl + 'shm_extension/variables';
            
            var requestData = JSON.stringify({ cells: cells });
            
            if (typeof fetch !== 'undefined') {
                fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: requestData
                })
                .then(response => response.json())
                .then(data => {
                    that.parsedVariables = data;
                    console.log('Server-parsed variables:', that.parsedVariables);
                    if (callback) callback(that.parsedVariables);
                })
                .catch(error => {
                    console.error('Failed to parse variables on server:', error);
                    // Fallback to client-side parsing
                    that.parseNotebookVariables();
                    if (callback) callback(that.parsedVariables);
                });
            } else if (typeof $ !== 'undefined') {
                $.ajax({
                    url: url,
                    method: 'POST',
                    contentType: 'application/json',
                    data: requestData,
                    success: function(data) {
                        that.parsedVariables = typeof data === 'string' ? JSON.parse(data) : data;
                        console.log('Server-parsed variables:', that.parsedVariables);
                        if (callback) callback(that.parsedVariables);
                    },
                    error: function(xhr, status, error) {
                        console.error('Failed to parse variables on server:', error);
                        // Fallback to client-side parsing
                        that.parseNotebookVariables();
                        if (callback) callback(that.parsedVariables);
                    }
                });
            } else {
                // No HTTP library available, use client-side parsing
                that.parseNotebookVariables();
                if (callback) callback(that.parsedVariables);
            }
        },
        
        // Extract variable assignments from code text
        extractVariablesFromCode: function(code, cellIndex) {
            var variables = [];
            var lines = code.split('\n');
            
            lines.forEach(function(line, lineIndex) {
                line = line.trim();
                
                // Skip comments and empty lines
                if (line.startsWith('#') || line === '') {
                    return;
                }
                
                // Look for assignment patterns
                var patterns = [
                    // Simple assignment: var = expression
                    /^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)/,
                    // Tuple unpacking: var1, var2 = expression
                    /^([a-zA-Z_][a-zA-Z0-9_,\s]*)\s*=\s*(.+)/,
                    // Multiple assignment with parentheses: (var1, var2) = expression  
                    /^\(([a-zA-Z_][a-zA-Z0-9_,\s]*)\)\s*=\s*(.+)/
                ];
                
                patterns.forEach(function(pattern) {
                    var match = line.match(pattern);
                    if (match) {
                        var leftSide = match[1].trim();
                        var rightSide = match[2].trim();
                        
                        // Handle tuple unpacking
                        if (leftSide.includes(',')) {
                            var varNames = leftSide.split(',');
                            varNames.forEach(function(varName) {
                                varName = varName.trim().replace(/[()]/g, '');
                                if (varName) {
                                    variables.push({
                                        name: varName,
                                        type: that.inferTypeFromExpression(rightSide),
                                        source: 'Cell ' + (cellIndex + 1),
                                        cellIndex: cellIndex,
                                        lineIndex: lineIndex,
                                        expression: rightSide
                                    });
                                }
                            });
                        } else {
                            // Single variable assignment
                            variables.push({
                                name: leftSide,
                                type: that.inferTypeFromExpression(rightSide),
                                source: 'Cell ' + (cellIndex + 1),
                                cellIndex: cellIndex,
                                lineIndex: lineIndex,
                                expression: rightSide
                            });
                        }
                    }
                });
            });
            
            return variables;
        },
        
        // Infer variable type from the right-hand side expression
        inferTypeFromExpression: function(expression) {
            // Remove comments from expression
            expression = expression.split('#')[0].trim();
            
            // Common SHM function patterns
            if (expression.includes('shmtools.')) {
                if (expression.includes('ar_model') || expression.includes('pca') || expression.includes('mahalanobis')) {
                    return 'tuple';  // Most SHM functions return multiple values
                }
                if (expression.includes('load_') || expression.includes('import_')) {
                    return 'numpy.ndarray';
                }
            }
            
            // NumPy patterns
            if (expression.includes('np.') || expression.includes('numpy.')) {
                if (expression.includes('.array') || expression.includes('.zeros') || expression.includes('.ones')) {
                    return 'numpy.ndarray';
                }
                if (expression.includes('.mean') || expression.includes('.std') || expression.includes('.sum')) {
                    return 'float';
                }
            }
            
            // Literal patterns
            if (/^\d+$/.test(expression)) {
                return 'int';
            }
            if (/^\d+\.\d+$/.test(expression)) {
                return 'float';
            }
            if (expression.startsWith('"') || expression.startsWith("'")) {
                return 'str';
            }
            if (expression.startsWith('[') && expression.endsWith(']')) {
                return 'list';
            }
            if (expression.startsWith('(') && expression.endsWith(')')) {
                return 'tuple';
            }
            if (expression.startsWith('{') && expression.endsWith('}')) {
                return 'dict';
            }
            
            // Default
            return 'unknown';
        }
    };
    
    // Return the extension object
    return {
        load_ipython_extension: SHMFunctionSelector.load_ipython_extension.bind(SHMFunctionSelector)
    };
}

// Auto-load for modern notebooks
if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {
    var extension = createSHMExtension(Jupyter);
    if (extension && extension.load_ipython_extension) {
        extension.load_ipython_extension();
    }
}