#!/usr/bin/env node

function parseParameters(parametersText) {
    const parameters = [];
    
    console.log("Original parameters text:");
    console.log(JSON.stringify(parametersText));
    
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

    console.log("Clean text after comment removal:");
    console.log(JSON.stringify(cleanText));
    
    // Now parse parameters from clean text
    // Fixed pattern: parameter_name = value (handles multi-line with proper lookahead)
    // The key fix: allow any whitespace (including newlines) between comma and next parameter
    const paramRegex = /(\w+)\s*=\s*([^,)]+?)(?=\s*,\s*|\s*$|\s*\))/g;
    let match;
    
    console.log("Using regex:", paramRegex);
    console.log("Testing regex matches:");
    
    // Reset regex lastIndex
    paramRegex.lastIndex = 0;
    
    while ((match = paramRegex.exec(cleanText)) !== null) {
        console.log(`Found match: "${match[0]}", param: "${match[1]}", value: "${match[2]}"`);
        
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
        
        // Find end of value
        let valueEnd = valueSearchStart;
        while (valueEnd < cleanText.length && cleanText[valueEnd] !== ',' && cleanText[valueEnd] !== ')') {
            valueEnd++;
        }
        
        // Map back to original positions
        const originalStart = positionMap[valueSearchStart] || 0;
        const originalEnd = positionMap[Math.min(valueEnd - 1, positionMap.length - 1)] || parametersText.length;
        
        parameters.push({
            name: paramName,
            value: paramValue,
            startPos: valueSearchStart,
            endPos: valueEnd
        });
        
        console.log(`Added parameter: ${paramName} = ${paramValue} (positions ${valueSearchStart}-${valueEnd})`);
    }
    
    console.log(`Total parameters found: ${parameters.length}`);
    return parameters;
}

// Test with the actual parameters text from the logs
const testParametersText = `
    X=data,  # array_like Input time series data of shape (TIME, CHANNELS, INSTANCES).
    ar_order=None,  # int, optional AR model order. (optional)
`;

console.log("=".repeat(80));
console.log("TESTING PARAMETER PARSING");
console.log("=".repeat(80));

const result = parseParameters(testParametersText);
console.log("Final result:", result);