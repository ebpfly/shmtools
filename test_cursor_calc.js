#!/usr/bin/env node
/**
 * Test the cursor position calculation logic from the extension
 */

function calculateAbsolutePosition(code, line, column) {
    const lines = code.split('\n');
    let absolutePos = 0;
    for (let i = 0; i < line; i++) {
        absolutePos += lines[i].length + 1; // +1 for newline
    }
    absolutePos += column;
    return absolutePos;
}

function findCursorPositionForNone(code) {
    const lines = code.split('\n');
    
    // Find the line containing "None"
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
        const line = lines[lineIndex];
        const nonePos = line.indexOf('None');
        if (nonePos !== -1) {
            // Position between "N" and "o"
            const columnPos = nonePos + 1;
            const absolutePos = calculateAbsolutePosition(code, lineIndex, columnPos);
            
            console.log(`Found "None" on line ${lineIndex}, column ${nonePos}`);
            console.log(`Cursor between N and o: line ${lineIndex}, column ${columnPos}`);
            console.log(`Absolute position: ${absolutePos}`);
            console.log(`Character at absolute position: "${code[absolutePos]}"`);
            
            return { line: lineIndex, column: columnPos, absolutePos };
        }
    }
    return null;
}

function extractFunctionCallAtPosition(code, cursorPos) {
    console.log(`🔍 Extracting function call at position ${cursorPos}`);
    console.log(`🔍 Code length: ${code.length}`);
    console.log(`🔍 Character at cursor: "${code[cursorPos] || 'END'}"`);
    
    // Strategy: Look for function patterns in the code and check if cursor is within their scope
    const functionPattern = /(\w+(?:\.\w+)*)\s*\(/g;
    let match;
    let bestMatch = null;
    const allMatches = [];
    
    // Find all function calls in the code
    while ((match = functionPattern.exec(code)) !== null) {
        const matchStart = match.index;
        const functionName = match[1];
        const openParenPos = match.index + match[0].length - 1;
        
        console.log(`🔍 Found function "${functionName}" at position ${matchStart}, opening paren at ${openParenPos}`);
        
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
        
        // Check if cursor is within this function call
        if (cursorPos >= matchStart && cursorPos <= functionEnd) {
            console.log(`🔍 Cursor is within function "${functionName}"!`);
            
            const fullText = code.substring(matchStart, functionEnd);
            const parametersText = code.substring(openParenPos + 1, functionEnd - 1);
            
            // Extract just the function name part (after the last dot)
            const nameParts = functionName.split('.');
            const simpleName = nameParts[nameParts.length - 1];
            
            const result = {
                functionName: simpleName,
                startPos: matchStart,
                endPos: functionEnd,
                fullText,
                parametersText
            };
            
            allMatches.push(result);
            
            // If we have multiple nested functions, prefer the innermost one
            if (!bestMatch || (matchStart > bestMatch.startPos)) {
                bestMatch = result;
            }
        }
    }
    
    return bestMatch;
}

function testCursorCalculation() {
    const testCode = `# Estimate autoregressive model parameters and compute RMSE.
ar_coeffs, rmse = shmtools.ar_model(
    X=data,  # array_like Input time series data of shape (TIME, CHANNELS, INSTANCES).
    ar_order=None,  # int, optional AR model order. (optional)
)`;
    
    console.log("=" .repeat(80));
    console.log("TEST CODE:");
    console.log(testCode);
    console.log("=" .repeat(80));
    console.log(`Total code length: ${testCode.length}`);
    console.log();
    
    // Find cursor position using the same logic as the extension
    const cursorInfo = findCursorPositionForNone(testCode);
    
    if (!cursorInfo) {
        console.log("❌ Could not find 'None' in code");
        return;
    }
    
    console.log();
    console.log("TESTING FUNCTION EXTRACTION:");
    console.log("=" .repeat(40));
    
    const result = extractFunctionCallAtPosition(testCode, cursorInfo.absolutePos);
    
    if (result) {
        console.log("✅ SUCCESS! Found function call:");
        console.log(`   Function: ${result.functionName}`);
        console.log(`   Range: ${result.startPos} - ${result.endPos}`);
        console.log(`   Parameters: '${result.parametersText}'`);
    } else {
        console.log("❌ FAILED! No function call found");
    }
}

if (require.main === module) {
    testCursorCalculation();
}