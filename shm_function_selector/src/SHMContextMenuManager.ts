// Context-menu / parameter-linking component for the SHM Function Selector
// extension.
//
// Handles right-click parameter detection, variable compatibility menus and
// plotting menus inside notebook code cells.
//
// Extracted from index.ts as part of the modular refactor (issue #30).

import { Cell } from '@jupyterlab/cells';

import { requestAPI } from './serverAPI';
import { ParameterContext, Variable } from './types';

export class SHMContextMenuManager {
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